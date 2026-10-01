#pragma once
#include "font8x8.h"
#include "webcam.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstdio>

// Per-frame state for the on-window HUD (fps readout, calibration banner).
struct HUDState {
    float fps = 0.0f;           // smoothed frames per second
    bool calibrating = false;   // calibration window active
    int calibProgress = 0;      // frames collected so far
    int calibTarget = 30;       // frames needed to complete
};

// Runtime-tunable knobs shown in the settings panel ([S]). The panel is part
// of the UI overlay and hides completely with the [W] visibility toggle.
struct SettingsState {
    // Head rotation gains / clamps (mirrored into RigSolver every frame)
    float headYawGain = 0.55f;
    float headPitchGain = 0.55f;
    float headRollGain = 0.55f;
    float headMaxYaw = 35.0f;   // degrees
    float headMaxPitch = 20.0f;
    float headMaxRoll = 15.0f;
    // 0 = responsive, 1 = very smooth (slightly more lag)
    float headSmoothing = 0.5f;
    // Same for the body pose path (arms, spine, lean)
    float bodySmoothing = 0.5f;
    // Eye gaze strength multiplier
    float gazeScale = 1.0f;
    // Springbones (hair/clothes physics)
    bool springEnabled = true;
    float springStiffness = 1.0f;
    float springGravity = 1.0f;
    // Background: 0=white 1=black 2=green (chroma key) 3=transparent
    int bgMode = 0;

    static constexpr int ROW_COUNT = 13;
    int selected = 0;

    static const char* bgModeName(int m) {
        switch (m) {
            case 1: return "Black";
            case 2: return "Green (key)";
            case 3: return "Transparent";
            default: return "White";
        }
    }
};

class UIOverlay {
public:
    bool visible = true;
    bool showCameraMenu = false;
    bool showSettings = false;
    SettingsState settings;
    std::string statusMessage;
    float statusTimer = 0.0f;
    uint8_t statusColor[3] = {100, 255, 120}; // Green default

    void toggleVisible() {
        visible = !visible;
        // Hiding the overlay hides everything it draws, including menus.
        if (!visible) {
            showCameraMenu = false;
            showSettings = false;
        }
    }

    void toggleCameraMenu() {
        showCameraMenu = !showCameraMenu;
        if (showCameraMenu) visible = true;
    }

    // Settings panel: only responds while the overlay is visible, so a
    // hidden UI ([W]) stays fully hidden.
    void toggleSettings() {
        if (!visible) return;
        showSettings = !showSettings;
    }

    void moveSettingSelection(int dir) {
        if (!showSettings) return;
        int n = SettingsState::ROW_COUNT;
        settings.selected = (settings.selected + dir + n) % n;
    }

    // dir: -1 (left/down), +1 (right/up)
    void adjustSetting(int dir) {
        if (!showSettings) return;
        SettingsState& s = settings;
        auto clampStep = [dir](float& v, float lo, float hi, float step) {
            v += dir * step;
            if (v < lo) v = lo;
            if (v > hi) v = hi;
        };
        switch (s.selected) {
            case 0: clampStep(s.headYawGain, 0.0f, 1.0f, 0.05f); break;
            case 1: clampStep(s.headPitchGain, 0.0f, 1.0f, 0.05f); break;
            case 2: clampStep(s.headRollGain, 0.0f, 1.0f, 0.05f); break;
            case 3: clampStep(s.headMaxYaw, 5.0f, 90.0f, 1.0f); break;
            case 4: clampStep(s.headMaxPitch, 5.0f, 60.0f, 1.0f); break;
            case 5: clampStep(s.headMaxRoll, 5.0f, 45.0f, 1.0f); break;
            case 6: clampStep(s.headSmoothing, 0.0f, 1.0f, 0.05f); break;
            case 7: clampStep(s.bodySmoothing, 0.0f, 1.0f, 0.05f); break;
            case 8: clampStep(s.gazeScale, 0.0f, 2.0f, 0.1f); break;
            case 9: s.springEnabled = !s.springEnabled; break;
            case 10: clampStep(s.springStiffness, 0.0f, 2.0f, 0.1f); break;
            case 11: clampStep(s.springGravity, 0.0f, 2.0f, 0.1f); break;
            case 12: s.bgMode = (s.bgMode + (dir > 0 ? 1 : 3)) % 4; break;
        }
    }

    void setStatus(const std::string& msg, float duration = 3.0f, bool isError = false) {
        statusMessage = msg;
        statusTimer = duration;
        if (isError) {
            statusColor[0] = 255; statusColor[1] = 90;  statusColor[2] = 90;  // Red
        } else {
            statusColor[0] = 100; statusColor[1] = 255; statusColor[2] = 120; // Green
        }
    }

    void update(float dt) {
        if (statusTimer > 0.0f) {
            statusTimer -= dt;
            if (statusTimer <= 0.0f) {
                statusMessage.clear();
            }
        }
    }

    static void drawPixelAlpha(uint8_t* bgra, int fbW, int fbH, int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
        if (x < 0 || x >= fbW || y < 0 || y >= fbH) return;
        int idx = (y * fbW + x) * 4;
        if (a == 255) {
            bgra[idx + 0] = 255;
            bgra[idx + 1] = r;
            bgra[idx + 2] = g;
            bgra[idx + 3] = b;
        } else {
            int invA = 255 - a;
            bgra[idx + 0] = 255;
            bgra[idx + 1] = (uint8_t)((r * a + bgra[idx + 1] * invA) / 255);
            bgra[idx + 2] = (uint8_t)((g * a + bgra[idx + 2] * invA) / 255);
            bgra[idx + 3] = (uint8_t)((b * a + bgra[idx + 3] * invA) / 255);
        }
    }

    static void drawBoxAlpha(uint8_t* bgra, int fbW, int fbH, int x, int y, int w, int h, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
        int x0 = std::max(0, x), x1 = std::min(fbW, x + w);
        int y0 = std::max(0, y), y1 = std::min(fbH, y + h);
        int invA = 255 - a;
        for (int cy = y0; cy < y1; cy++) {
            for (int cx = x0; cx < x1; cx++) {
                int idx = (cy * fbW + cx) * 4;
                bgra[idx + 0] = 255;
                bgra[idx + 1] = (uint8_t)((r * a + bgra[idx + 1] * invA) / 255);
                bgra[idx + 2] = (uint8_t)((g * a + bgra[idx + 2] * invA) / 255);
                bgra[idx + 3] = (uint8_t)((b * a + bgra[idx + 3] * invA) / 255);
            }
        }
    }

    static void drawBorder(uint8_t* bgra, int fbW, int fbH, int x, int y, int w, int h, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
        for (int cx = x; cx < x + w; cx++) {
            drawPixelAlpha(bgra, fbW, fbH, cx, y, r, g, b, a);
            drawPixelAlpha(bgra, fbW, fbH, cx, y + h - 1, r, g, b, a);
        }
        for (int cy = y; cy < y + h; cy++) {
            drawPixelAlpha(bgra, fbW, fbH, x, cy, r, g, b, a);
            drawPixelAlpha(bgra, fbW, fbH, x + w - 1, cy, r, g, b, a);
        }
    }

    static void drawChar(uint8_t* bgra, int fbW, int fbH, int x, int y, char c, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255, int scale = 1) {
        uint8_t uc = (uint8_t)c;
        if (uc > 127) uc = '?';
        const uint8_t* bitmap = font8x8_basic[uc];
        for (int row = 0; row < 8; row++) {
            uint8_t byte = bitmap[row];
            for (int col = 0; col < 8; col++) {
                if (byte & (1 << col)) {
                    if (scale == 1) {
                        drawPixelAlpha(bgra, fbW, fbH, x + col, y + row, r, g, b, a);
                    } else {
                        for (int sy = 0; sy < scale; sy++) {
                            for (int sx = 0; sx < scale; sx++) {
                                drawPixelAlpha(bgra, fbW, fbH, x + col * scale + sx, y + row * scale + sy, r, g, b, a);
                            }
                        }
                    }
                }
            }
        }
    }

    static void drawText(uint8_t* bgra, int fbW, int fbH, int x, int y, const std::string& text, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255, int scale = 1) {
        int curX = x;
        for (char c : text) {
            if (c == '\n') {
                curX = x;
                y += 10 * scale;
                continue;
            }
            drawChar(bgra, fbW, fbH, curX, y, c, r, g, b, a, scale);
            curX += 8 * scale;
        }
    }

    void render(uint8_t* bgra, int fbW, int fbH, const std::vector<CameraDeviceInfo>& cameras, SDL_CameraID activeDevId, const std::string& activeName, const HUDState& hud) {
        if (!visible) return;

        // 0. FPS badge at top-right
        {
            std::string fpsStr = std::to_string((int)(hud.fps + 0.5f)) + " fps";
            if (hud.fps < 0.5f) fpsStr = "-- fps";
            int badgeW = (int)fpsStr.length() * 8 + 20;
            int badgeX = fbW - 14 - badgeW;
            drawBoxAlpha(bgra, fbW, fbH, badgeX, 14, badgeW, 26, 18, 22, 28, 215);
            drawBorder(bgra, fbW, fbH, badgeX, 14, badgeW, 26, 60, 75, 95, 200);
            drawText(bgra, fbW, fbH, badgeX + 10, 23, fpsStr, 255, 205, 80, 255);
        }

        // 1. Compact HUD bar at Top-Left
        const int hudX = 14;
        const int hudY = 14;
        const int hudH = 26;
        
        std::string camLabel = " Camera: " + activeName;
        if (camLabel.size() > 32) camLabel = camLabel.substr(0, 29) + "...";
        std::string hudText = "[C]" + camLabel + "  |  [W] PiP/UI  [S] Settings  [SPACE] Calib  [ESC] Quit";
        int hudW = (int)hudText.length() * 8 + 20;

        // Draw HUD background & border
        drawBoxAlpha(bgra, fbW, fbH, hudX, hudY, hudW, hudH, 18, 22, 28, 215);
        drawBorder(bgra, fbW, fbH, hudX, hudY, hudW, hudH, 60, 75, 95, 200);

        // Draw HUD text with highlighted [C]
        drawText(bgra, fbW, fbH, hudX + 10, hudY + 9, "[C]", 80, 200, 255, 255);
        drawText(bgra, fbW, fbH, hudX + 10 + 24, hudY + 9, camLabel, 240, 245, 255, 255);
        int afterCamX = hudX + 10 + 24 + (int)camLabel.length() * 8;
        drawText(bgra, fbW, fbH, afterCamX, hudY + 9, "  |  [W] PiP/UI  [S] Settings  [SPACE] Calib  [ESC] Quit", 160, 175, 195, 255);

        int currentY = hudY + hudH + 8;

        // 2. Calibration banner (while the calibration window is active)
        if (hud.calibrating) {
            std::string msg = (hud.calibProgress > 0)
                ? "Calibrating... " + std::to_string(hud.calibProgress) + "/" + std::to_string(hud.calibTarget)
                : "Calibrating... (waiting for face)";
            const int banW = 244;
            const int banH = 42;
            drawBoxAlpha(bgra, fbW, fbH, hudX, currentY, banW, banH, 32, 25, 12, 235);
            drawBorder(bgra, fbW, fbH, hudX, currentY, banW, banH, 255, 205, 80, 220);
            drawText(bgra, fbW, fbH, hudX + 12, currentY + 7, msg, 255, 225, 150, 255);
            // Progress bar
            int barX = hudX + 12, barY = currentY + 26, barW = banW - 24, barH = 6;
            drawBoxAlpha(bgra, fbW, fbH, barX, barY, barW, barH, 55, 45, 28, 255);
            float frac = (hud.calibTarget > 0)
                ? std::min(1.0f, (float)hud.calibProgress / (float)hud.calibTarget) : 0.0f;
            int fillW = (int)((barW - 2) * frac);
            if (fillW > 0)
                drawBoxAlpha(bgra, fbW, fbH, barX + 1, barY + 1, fillW, barH - 2, 255, 205, 80, 255);
            currentY += banH + 8;
        }

        // 3. Status message badge (if active)
        if (!statusMessage.empty() && statusTimer > 0.0f) {
            int statW = (int)statusMessage.length() * 8 + 24;
            int statH = 24;
            drawBoxAlpha(bgra, fbW, fbH, hudX, currentY, statW, statH, 15, 30, 24, 230);
            drawBorder(bgra, fbW, fbH, hudX, currentY, statW, statH, statusColor[0], statusColor[1], statusColor[2], 220);
            drawText(bgra, fbW, fbH, hudX + 12, currentY + 8, statusMessage, statusColor[0], statusColor[1], statusColor[2], 255);
            currentY += statH + 8;
        }

        // 4. Expanded Camera Menu & Shortcuts Legend (when showCameraMenu is true)
        if (showCameraMenu) {
            const int menuW = 500;
            int camCount = (int)cameras.size();
            int numCamRows = std::max(1, camCount);
            // Header (30px) + Cameras (numCamRows * 20px + 10px) + Divider (10px) + Legend (7 rows * 18px + 15px)
            int menuH = 34 + (numCamRows * 22) + 12 + 150;

            // Background & glassmorphic border
            drawBoxAlpha(bgra, fbW, fbH, hudX, currentY, menuW, menuH, 14, 18, 24, 235);
            drawBorder(bgra, fbW, fbH, hudX, currentY, menuW, menuH, 70, 95, 130, 240);

            // Menu Header
            drawBoxAlpha(bgra, fbW, fbH, hudX + 1, currentY + 1, menuW - 2, 28, 26, 34, 46, 250);
            drawText(bgra, fbW, fbH, hudX + 14, currentY + 10, "CAMERA SOURCES", 80, 210, 255, 255);
            drawText(bgra, fbW, fbH, hudX + menuW - 100, currentY + 10, "[C] Close", 150, 165, 180, 255);

            int itemY = currentY + 36;
            if (cameras.empty()) {
                drawText(bgra, fbW, fbH, hudX + 16, itemY + 4, "No cameras detected. Plug in camera & press [R]", 255, 120, 120, 255);
                itemY += 24;
            } else {
                for (int i = 0; i < camCount; i++) {
                    bool isActive = (cameras[i].id == activeDevId);
                    if (isActive) {
                        // Highlight active row
                        drawBoxAlpha(bgra, fbW, fbH, hudX + 6, itemY, menuW - 12, 20, 30, 60, 90, 200);
                        drawBorder(bgra, fbW, fbH, hudX + 6, itemY, menuW - 12, 20, 60, 140, 220, 200);
                    }

                    std::string keyPrefix = "[" + std::to_string(i + 1) + "]";
                    if (i >= 9) keyPrefix = "   "; // keys 1-9 direct switch

                    std::string name = cameras[i].name;
                    if (name.size() > 42) name = name.substr(0, 39) + "...";
                    if (isActive) name += " (Active)";

                    if (isActive) {
                        drawText(bgra, fbW, fbH, hudX + 14, itemY + 6, keyPrefix, 100, 255, 150, 255);
                        drawText(bgra, fbW, fbH, hudX + 46, itemY + 6, name, 255, 255, 255, 255);
                    } else {
                        drawText(bgra, fbW, fbH, hudX + 14, itemY + 6, keyPrefix, 100, 180, 240, 255);
                        drawText(bgra, fbW, fbH, hudX + 46, itemY + 6, name, 200, 210, 220, 255);
                    }
                    itemY += 22;
                }
            }

            itemY += 8;
            // Divider line
            drawBoxAlpha(bgra, fbW, fbH, hudX + 10, itemY, menuW - 20, 1, 60, 75, 100, 220);
            itemY += 10;

            // Shortcuts Legend Title
            drawText(bgra, fbW, fbH, hudX + 14, itemY, "KEYBOARD SHORTCUTS LEGEND:", 255, 205, 80, 255);
            itemY += 18;

            auto drawLegendLine = [&](const std::string& key, const std::string& desc) {
                drawText(bgra, fbW, fbH, hudX + 18, itemY, key, 100, 220, 255, 255);
                drawText(bgra, fbW, fbH, hudX + 115, itemY, ": " + desc, 210, 220, 230, 255);
                itemY += 17;
            };

            std::string camKeyRange = (camCount > 1) ? ("[1]-[" + std::to_string(std::min(camCount, 9)) + "]") : "[1]";
            drawLegendLine(camKeyRange, "Select active camera source");
            drawLegendLine("[R]", "Rescan connected video devices");
            drawLegendLine("[C]", "Toggle camera menu & shortcuts");
            drawLegendLine("[S]", "Toggle settings (head/gaze/spring/bg)");
            drawLegendLine("[W]", "Toggle PiP preview & UI overlay");
            drawLegendLine("[SPACE]", "Calibrate / re-calibrate tracking pose");
            drawLegendLine("[P]", "Save screenshot (PNG, alpha in bg=Transparent)");
            drawLegendLine("[ESC]", "Quit application");
        }

        // 5. Settings panel (when showSettings is true). Part of the overlay:
        // hidden entirely when the [W] visibility toggle is off.
        if (showSettings) {
            const SettingsState& st = settings;
            const int panelW = 470;
            const int rowH = 24;
            const int panelH = 34 + SettingsState::ROW_COUNT * rowH + 10;

            drawBoxAlpha(bgra, fbW, fbH, hudX, currentY, panelW, panelH, 14, 18, 24, 235);
            drawBorder(bgra, fbW, fbH, hudX, currentY, panelW, panelH, 70, 95, 130, 240);
            drawBoxAlpha(bgra, fbW, fbH, hudX + 1, currentY + 1, panelW - 2, 28, 26, 34, 46, 250);
            drawText(bgra, fbW, fbH, hudX + 14, currentY + 10, "SETTINGS", 80, 210, 255, 255);
            drawText(bgra, fbW, fbH, hudX + 150, currentY + 10,
                     "UP/DOWN select   LEFT/RIGHT adjust", 150, 165, 180, 255);

            // Row descriptors: label + value text + slider fraction (-1 = none)
            struct Row { const char* label; std::string value; float frac; };
            auto rows = [] (const SettingsState& st) {
                std::vector<Row> r;
                auto f2 = [](float v) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "%.2f", v);
                    return std::string(buf);
                };
                r.push_back({"Head yaw gain", f2(st.headYawGain), st.headYawGain});
                r.push_back({"Head pitch gain", f2(st.headPitchGain), st.headPitchGain});
                r.push_back({"Head roll gain", f2(st.headRollGain), st.headRollGain});
                r.push_back({"Head max yaw (deg)", f2(st.headMaxYaw), (st.headMaxYaw - 5) / 85.0f});
                r.push_back({"Head max pitch (deg)", f2(st.headMaxPitch), (st.headMaxPitch - 5) / 55.0f});
                r.push_back({"Head max roll (deg)", f2(st.headMaxRoll), (st.headMaxRoll - 5) / 40.0f});
                r.push_back({"Head smoothing", f2(st.headSmoothing), st.headSmoothing});
                r.push_back({"Body smoothing", f2(st.bodySmoothing), st.bodySmoothing});
                r.push_back({"Gaze scale", f2(st.gazeScale), st.gazeScale / 2.0f});
                r.push_back({"Springbones", std::string(st.springEnabled ? "ON" : "OFF"), -1.0f});
                r.push_back({"Spring stiffness", f2(st.springStiffness), st.springStiffness / 2.0f});
                r.push_back({"Spring gravity", f2(st.springGravity), st.springGravity / 2.0f});
                r.push_back({"Background", SettingsState::bgModeName(st.bgMode), -1.0f});
                return r;
            };
            std::vector<Row> rowList = rows(st);

            int itemY = currentY + 36;
            const int sliderX = hudX + panelW - 130;
            const int sliderW = 110;
            for (int i = 0; i < (int)rowList.size() && i < SettingsState::ROW_COUNT; i++) {
                const Row& row = rowList[i];
                bool isSel = (i == st.selected);
                if (isSel) {
                    drawBoxAlpha(bgra, fbW, fbH, hudX + 6, itemY, panelW - 12, rowH - 4, 30, 60, 90, 200);
                    drawBorder(bgra, fbW, fbH, hudX + 6, itemY, panelW - 12, rowH - 4, 60, 140, 220, 200);
                }
                drawText(bgra, fbW, fbH, hudX + 14, itemY + 5,
                         isSel ? "> " + std::string(row.label) : std::string("  ") + row.label,
                         isSel ? 255 : 200, isSel ? 255 : 210, isSel ? 255 : 220, 255);
                // Value + slider
                drawText(bgra, fbW, fbH, hudX + 195, itemY + 5, row.value, 255, 205, 80, 255);
                if (row.frac >= 0.0f) {
                    drawBoxAlpha(bgra, fbW, fbH, sliderX, itemY + 8, sliderW, 6, 55, 45, 28, 255);
                    int fillW = (int)((sliderW - 2) * row.frac);
                    if (fillW > 0)
                        drawBoxAlpha(bgra, fbW, fbH, sliderX + 1, itemY + 9, fillW, 4, 255, 205, 80, 255);
                } else {
                    // Toggle / cycle marker
                    const char* arrows = "< >";
                    drawText(bgra, fbW, fbH, sliderX + sliderW / 2 - 12, itemY + 5, arrows, 120, 135, 155, 255);
                }
                itemY += rowH;
            }
        }
    }
};
