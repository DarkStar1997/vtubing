#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Status ---- */
typedef enum MpStatus {
    kMpOk = 0,
    kMpCancelled = 1,
    kMpUnknown = 2,
    kMpInvalidArgument = 3,
    kMpDeadlineExceeded = 4,
    kMpNotFound = 5,
    kMpAlreadyExists = 6,
    kMpPermissionDenied = 7,
    kMpResourceExhausted = 8,
    kMpFailedPrecondition = 9,
    kMpAborted = 10,
    kMpOutOfRange = 11,
    kMpUnimplemented = 12,
    kMpInternal = 13,
    kMpUnavailable = 14,
    kMpDataLoss = 15,
    kMpUnauthenticated = 16,
} MpStatus;

/* ---- Running mode ---- */
enum MpRunningMode {
    MP_RUNNING_MODE_IMAGE = 1,
    MP_RUNNING_MODE_VIDEO = 2,
    MP_RUNNING_MODE_LIVE_STREAM = 3,
};

/* ---- Image ---- */
typedef enum MpImageFormat {
    kMpImageFormatUnknown = 0,
    kMpImageFormatSrgb = 1,
    kMpImageFormatSrgba = 2,
    kMpImageFormatGray8 = 3,
    kMpImageFormatGray16 = 4,
    kMpImageFormatSrgb48 = 7,
    kMpImageFormatSrgba64 = 8,
    kMpImageFormatVec32F1 = 9,
    kMpImageFormatVec32F2 = 12,
    kMpImageFormatVec32F4 = 13,
} MpImageFormat;

typedef struct MpImageInternal* MpImagePtr;

MpStatus MpImageCreateFromUint8Data(
    MpImageFormat format, int width, int height,
    const uint8_t* pixel_data, int pixel_data_size,
    MpImagePtr* out, char** error_msg);

void MpImageFree(MpImagePtr image);

/* ---- Landmarks ---- */
struct MpLandmark {
    float x, y, z;
    bool has_visibility;
    float visibility;
    bool has_presence;
    float presence;
    char* name;
};

struct MpNormalizedLandmark {
    float x, y, z;
    bool has_visibility;
    float visibility;
    bool has_presence;
    float presence;
    char* name;
};

struct MpLandmarks {
    struct MpLandmark* landmarks;
    uint32_t landmarks_count;
};

struct MpNormalizedLandmarks {
    struct MpNormalizedLandmark* landmarks;
    uint32_t landmarks_count;
};

/* ---- Base options ---- */
enum MpDelegate {
    MP_DELEGATE_CPU = 0,
    MP_DELEGATE_GPU = 1,
    MP_DELEGATE_EDGETPU_NNAPI = 2,
};

enum MpHostEnvironment {
    MP_HOST_ENVIRONMENT_UNKNOWN = 0,
    MP_HOST_ENVIRONMENT_ANDROID = 1,
    MP_HOST_ENVIRONMENT_IOS = 2,
    MP_HOST_ENVIRONMENT_PYTHON = 3,
    MP_HOST_ENVIRONMENT_WEB = 4,
};

enum MpHostSystem {
    MP_HOST_SYSTEM_UNKNOWN = 0,
    MP_HOST_SYSTEM_LINUX = 1,
    MP_HOST_SYSTEM_MAC = 2,
    MP_HOST_SYSTEM_WINDOWS = 3,
    MP_HOST_SYSTEM_IOS = 4,
    MP_HOST_SYSTEM_ANDROID = 5,
};

struct MpBaseOptions {
    const char* model_asset_buffer;
    unsigned int model_asset_buffer_count;
    const char* model_asset_path;
    enum MpDelegate delegate;
    enum MpHostEnvironment host_environment;
    enum MpHostSystem host_system;
    const char* host_version;
    const char* ca_bundle_path;
};

/* ---- Image processing options (forward-declared, we pass NULL) ---- */
typedef struct MpImageProcessingOptions MpImageProcessingOptions;

/* ---- Pose landmarker ---- */
typedef struct MpPoseLandmarkerInternal* MpPoseLandmarkerPtr;

struct MpPoseLandmarkerOptions {
    struct MpBaseOptions base_options;
    MpRunningMode running_mode;
    int num_poses;
    float min_pose_detection_confidence;
    float min_pose_presence_confidence;
    float min_tracking_confidence;
    bool output_segmentation_masks;
    typedef void (*result_callback_fn)(MpStatus status,
                                       const struct MpPoseLandmarkerResult* result,
                                       const MpImagePtr image,
                                       int64_t timestamp_ms);
    result_callback_fn result_callback;
};

struct MpPoseLandmarkerResult {
    MpImagePtr* segmentation_masks;
    uint32_t segmentation_masks_count;
    struct MpNormalizedLandmarks* pose_landmarks;
    uint32_t pose_landmarks_count;
    struct MpLandmarks* pose_world_landmarks;
    uint32_t pose_world_landmarks_count;
};

MpStatus MpPoseLandmarkerCreate(
    struct MpPoseLandmarkerOptions* options,
    MpPoseLandmarkerPtr* landmarker_out, char** error_msg);

MpStatus MpPoseLandmarkerDetectForVideo(
    MpPoseLandmarkerPtr landmarker, MpImagePtr image,
    const MpImageProcessingOptions* options, int64_t timestamp_ms,
    struct MpPoseLandmarkerResult* result, char** error_msg);

void MpPoseLandmarkerCloseResult(struct MpPoseLandmarkerResult* result);

MpStatus MpPoseLandmarkerClose(MpPoseLandmarkerPtr landmarker, char** error_msg);

/* ---- Category (classification output) ---- */
struct MpCategory {
    int index;
    float score;
    char* category_name;
    char* display_name;
};

struct MpCategories {
    struct MpCategory* categories;
    uint32_t categories_count;
};

/* ---- Matrix (column-major, OpenGL convention) ---- */
struct MpMatrix {
    uint32_t rows;
    uint32_t cols;
    float* data;
};

/* ---- Face landmarker ---- */
typedef struct MpFaceLandmarkerInternal* MpFaceLandmarkerPtr;

struct MpFaceLandmarkerOptions {
    struct MpBaseOptions base_options;
    MpRunningMode running_mode;
    int num_faces;
    float min_face_detection_confidence;
    float min_face_presence_confidence;
    float min_tracking_confidence;
    bool output_face_blendshapes;
    bool output_facial_transformation_matrixes;
    typedef void (*result_callback_fn)(MpStatus status,
                                       const struct MpFaceLandmarkerResult* result,
                                       const MpImagePtr image,
                                       int64_t timestamp_ms);
    result_callback_fn result_callback;
};

struct MpFaceLandmarkerResult {
    struct MpNormalizedLandmarks* face_landmarks;
    uint32_t face_landmarks_count;
    struct MpCategories* face_blendshapes;
    uint32_t face_blendshapes_count;
    struct MpMatrix* facial_transformation_matrixes;
    uint32_t facial_transformation_matrixes_count;
};

MpStatus MpFaceLandmarkerCreate(
    struct MpFaceLandmarkerOptions* options,
    MpFaceLandmarkerPtr* landmarker_out, char** error_msg);

MpStatus MpFaceLandmarkerDetectForVideo(
    MpFaceLandmarkerPtr landmarker, MpImagePtr image,
    const MpImageProcessingOptions* options, int64_t timestamp_ms,
    struct MpFaceLandmarkerResult* result, char** error_msg);

void MpFaceLandmarkerCloseResult(struct MpFaceLandmarkerResult* result);

MpStatus MpFaceLandmarkerClose(MpFaceLandmarkerPtr landmarker, char** error_msg);

/* ---- Hand landmarker ---- */
typedef struct MpHandLandmarkerInternal* MpHandLandmarkerPtr;

struct MpHandLandmarkerOptions {
    struct MpBaseOptions base_options;
    MpRunningMode running_mode;
    int num_hands;
    float min_hand_detection_confidence;
    float min_hand_presence_confidence;
    float min_tracking_confidence;
    typedef void (*result_callback_fn)(MpStatus status,
                                       const struct MpHandLandmarkerResult* result,
                                       const MpImagePtr image,
                                       int64_t timestamp_ms);
    result_callback_fn result_callback;
};

struct MpHandLandmarkerResult {
    struct MpCategories* handedness;
    uint32_t handedness_count;
    struct MpNormalizedLandmarks* hand_landmarks;
    uint32_t hand_landmarks_count;
    struct MpLandmarks* hand_world_landmarks;
    uint32_t hand_world_landmarks_count;
};

MpStatus MpHandLandmarkerCreate(
    struct MpHandLandmarkerOptions* options,
    MpHandLandmarkerPtr* landmarker_out, char** error_msg);

MpStatus MpHandLandmarkerDetectForVideo(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    const MpImageProcessingOptions* options, int64_t timestamp_ms,
    struct MpHandLandmarkerResult* result, char** error_msg);

void MpHandLandmarkerCloseResult(struct MpHandLandmarkerResult* result);

MpStatus MpHandLandmarkerClose(MpHandLandmarkerPtr landmarker, char** error_msg);

void MpErrorFree(char* error_message);

#ifdef __cplusplus
}
#endif
