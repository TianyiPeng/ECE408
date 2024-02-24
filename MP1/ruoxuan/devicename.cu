#include <stdio.h>
#include <stdlib.h>
#include <nvml.h>

int main() {
    unsigned int devicesCount;
    int i;
    char deviceName[256];
    nvmlReturn_t result;

    // 初始化NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to initialize NVML: %d\n", result);
        return EXIT_FAILURE;
    }

    // 获取GPU数量
    result = nvmlDeviceGetCount(&devicesCount);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to get device count: %d\n", result);
        nvmlShutdown();
        return EXIT_FAILURE;
    }

    // 遍历所有GPU
    for (i = 0; i < devicesCount; i++) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result != NVML_SUCCESS) {
            fprintf(stderr, "Failed to get device handle for index %d: %d\n", i, result);
            continue;
        }
	
        // 获取GPU名称
        result = nvmlDeviceGetName(device, deviceName, sizeof(deviceName));
        if (result != NVML_SUCCESS) {
            fprintf(stderr, "Failed to get device name for index %d: %d\n", i, result);
            continue;
        }

        // 打印GPU名称
        printf("Device %d: %s\n", i, deviceName);

        // 尝试区分A100 SXM和PCIe版本
        if (strstr(deviceName, "A100-SXM") != NULL) {
            printf("  This is an A100 SXM GPU.\n");
        } else if (strstr(deviceName, "A100-PCIe") != NULL) {
            printf("  This is an A100 PCIe GPU.\n");
        } else {
            printf("  Model not recognized.\n");
        }
    }

    // 清理NVML
    nvmlShutdown();

    return EXIT_SUCCESS;
}

