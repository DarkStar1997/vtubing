#pragma once
#include "font8x8.h"
#include "webcam.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>

class UIOverlay {
public:
    bool visible = true;
    bool showCameraMenu = false;
    std::string statusMessage;
    float statusTimer = 0.0f;
    uint8_t statusColor[3] = {100, 255, 120}; // Green default

    void toggleVisible() {
        visible = !visible;
    }

    void toggleCameraMenu() {
        showCameraMenu = !showCameraMenu;
        if (showCameraMenu) visible = true;
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

    void render(uint8_t* bgra, int fbW, int fbH, const std::vector<CameraDeviceInfo>& cameras, SDL_CameraID activeDevId, const std::string& activeName) {
        if (!visible) return;

        // 1. Compact HUD bar at Top-Left
        const int hudX = 14;
        const int hudY = 14;
        const int hudH = 26;
        
        std::string camLabel = " Camera: " + activeName;
        if (camLabel.size() > 32) camLabel = camLabel.substr(0, 29) + "...";
        std::string hudText = "[C]" + camLabel + "  |  [W] PiP/UI  [SPACE] Calib  [ESC] Quit";
        int hudW = (int)hudText.length() * 8 + 20;

        // Draw HUD background & border
        drawBoxAlpha(bgra, fbW, fbH, hudX, hudY, hudW, hudH, 18, 22, 28, 215);
        drawBorder(bgra, fbW, fbH, hudX, hudY, hudW, hudH, 60, 75, 95, 200);

        // Draw HUD text with highlighted [C]
        drawText(bgra, fbW, fbH, hudX + 10, hudY + 9, "[C]", 80, 200, 255, 255);
        drawText(bgra, fbW, fbH, hudX + 10 + 24, hudY + 9, camLabel, 240, 245, 255, 255);
        int afterCamX = hudX + 10 + 24 + (int)camLabel.length() * 8;
        drawText(bgra, fbW, fbH, afterCamX, hudY + 9, "  |  [W] PiP/UI  [SPACE] Calib  [ESC] Quit", 160, 175, 195, 255);

        int currentY = hudY + hudH + 8;

        // 2. Status message badge (if active)
        if (!statusMessage.empty() && statusTimer > 0.0f) {
            int statW = (int)statusMessage.length() * 8 + 24;
            int statH = 24;
            drawBoxAlpha(bgra, fbW, fbH, hudX, currentY, statW, statH, 15, 30, 24, 230);
            drawBorder(bgra, fbW, fbH, hudX, currentY, statW, statH, statusColor[0], statusColor[1], statusColor[2], 220);
            drawText(bgra, fbW, fbH, hudX + 12, currentY + 8, statusMessage, statusColor[0], statusColor[1], statusColor[2], 255);
            currentY += statH + 8;
        }

        // 3. Expanded Camera Menu & Shortcuts Legend (when showCameraMenu is true)
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
            drawLegendLine("[W]", "Toggle PiP preview & UI overlay");
            drawLegendLine("[SPACE]", "Calibrate neutral tracking pose");
            drawLegendLine("[ESC]", "Quit application");
        }
    }
};
