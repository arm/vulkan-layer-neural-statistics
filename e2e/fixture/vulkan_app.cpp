/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
 * SPDX-License-Identifier: MIT
 */

#include "vulkan_app.hpp"
#include "embedded_assets.hpp"
#include "events.hpp"
#include "graph_assets.hpp"
#include "topology_plan.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace fixture
{
namespace
{

class VulkanError final : public std::runtime_error
{
  public:
    VulkanError(std::string operation, VkResult result)
        : std::runtime_error(std::move(operation) + " failed with VkResult " +
                             std::to_string(result)),
          result_(result)
    {
    }
    VkResult result() const
    {
        return result_;
    }

  private:
    VkResult result_;
};

void check(VkResult result, const char *operation)
{
    if (result != VK_SUCCESS)
        throw VulkanError(operation, result);
}

template <typename T> T requireProc(T proc, const char *name)
{
    if (!proc)
        throw std::runtime_error(std::string("Vulkan loader did not expose ") + name);
    return proc;
}

bool containsExtension(const std::vector<VkExtensionProperties> &extensions, const char *name)
{
    return std::any_of(extensions.begin(), extensions.end(),
                       [name](const VkExtensionProperties &extension)
                       { return std::strcmp(extension.extensionName, name) == 0; });
}

struct DispatchTable
{
    void *library{};
    PFN_vkGetInstanceProcAddr getInstanceProcAddr{};
    PFN_vkCreateInstance createInstance{};
    PFN_vkDestroyInstance destroyInstance{};
    PFN_vkEnumeratePhysicalDevices enumeratePhysicalDevices{};
    PFN_vkGetPhysicalDeviceQueueFamilyProperties getPhysicalDeviceQueueFamilyProperties{};
    PFN_vkGetPhysicalDeviceProperties getPhysicalDeviceProperties{};
    PFN_vkEnumerateDeviceExtensionProperties enumerateDeviceExtensionProperties{};
    PFN_vkGetPhysicalDeviceFeatures2 getPhysicalDeviceFeatures2{};
    PFN_vkGetPhysicalDeviceMemoryProperties getPhysicalDeviceMemoryProperties{};
    PFN_vkCreateDevice createDevice{};
    PFN_vkGetDeviceProcAddr getDeviceProcAddr{};

    PFN_vkDestroyDevice destroyDevice{};
    PFN_vkGetDeviceQueue getDeviceQueue{};
    PFN_vkDeviceWaitIdle deviceWaitIdle{};
    PFN_vkCreateTensorARM createTensor{};
    PFN_vkDestroyTensorARM destroyTensor{};
    PFN_vkGetTensorMemoryRequirementsARM getTensorMemoryRequirements{};
    PFN_vkBindTensorMemoryARM bindTensorMemory{};
    PFN_vkCreateTensorViewARM createTensorView{};
    PFN_vkDestroyTensorViewARM destroyTensorView{};
    PFN_vkAllocateMemory allocateMemory{};
    PFN_vkFreeMemory freeMemory{};
    PFN_vkCreateBuffer createBuffer{};
    PFN_vkDestroyBuffer destroyBuffer{};
    PFN_vkGetBufferMemoryRequirements getBufferMemoryRequirements{};
    PFN_vkBindBufferMemory bindBufferMemory{};
    PFN_vkMapMemory mapMemory{};
    PFN_vkUnmapMemory unmapMemory{};
    PFN_vkFlushMappedMemoryRanges flushMappedMemoryRanges{};
    PFN_vkCreateDescriptorSetLayout createDescriptorSetLayout{};
    PFN_vkDestroyDescriptorSetLayout destroyDescriptorSetLayout{};
    PFN_vkCreatePipelineLayout createPipelineLayout{};
    PFN_vkDestroyPipelineLayout destroyPipelineLayout{};
    PFN_vkCreateShaderModule createShaderModule{};
    PFN_vkDestroyShaderModule destroyShaderModule{};
    PFN_vkCreateDataGraphPipelinesARM createDataGraphPipelines{};
    PFN_vkDestroyPipeline destroyPipeline{};
    PFN_vkCreateDataGraphPipelineSessionARM createDataGraphPipelineSession{};
    PFN_vkDestroyDataGraphPipelineSessionARM destroyDataGraphPipelineSession{};
    PFN_vkGetDataGraphPipelineSessionBindPointRequirementsARM getSessionBindPointRequirements{};
    PFN_vkGetDataGraphPipelineSessionMemoryRequirementsARM getSessionMemoryRequirements{};
    PFN_vkBindDataGraphPipelineSessionMemoryARM bindSessionMemory{};
    PFN_vkCreateDescriptorPool createDescriptorPool{};
    PFN_vkDestroyDescriptorPool destroyDescriptorPool{};
    PFN_vkAllocateDescriptorSets allocateDescriptorSets{};
    PFN_vkUpdateDescriptorSets updateDescriptorSets{};
    PFN_vkCreateCommandPool createCommandPool{};
    PFN_vkDestroyCommandPool destroyCommandPool{};
    PFN_vkAllocateCommandBuffers allocateCommandBuffers{};
    PFN_vkBeginCommandBuffer beginCommandBuffer{};
    PFN_vkEndCommandBuffer endCommandBuffer{};
    PFN_vkCmdBindPipeline cmdBindPipeline{};
    PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets{};
    PFN_vkCmdDispatchDataGraphARM cmdDispatchDataGraph{};
    PFN_vkCmdBeginConditionalRenderingEXT cmdBeginConditionalRendering{};
    PFN_vkCmdEndConditionalRenderingEXT cmdEndConditionalRendering{};
    PFN_vkCmdExecuteCommands cmdExecuteCommands{};
    PFN_vkCmdCopyTensorARM cmdCopyTensor{};
    PFN_vkCmdPipelineBarrier2 cmdPipelineBarrier2{};
    PFN_vkCmdPipelineBarrier2KHR cmdPipelineBarrier2KHR{};
    PFN_vkCreateFence createFence{};
    PFN_vkDestroyFence destroyFence{};
    PFN_vkResetFences resetFences{};
    PFN_vkWaitForFences waitForFences{};
    PFN_vkGetFenceStatus getFenceStatus{};
    PFN_vkQueueSubmit queueSubmit{};
    PFN_vkQueueSubmit2 queueSubmit2{};
    PFN_vkQueueSubmit2KHR queueSubmit2KHR{};
    PFN_vkCreateSemaphore createSemaphore{};
    PFN_vkDestroySemaphore destroySemaphore{};

    ~DispatchTable()
    {
        if (library)
            dlclose(library);
    }

    template <typename T> T instanceProc(VkInstance instance, const char *name)
    {
        return reinterpret_cast<T>(getInstanceProcAddr(instance, name));
    }
    template <typename T> T deviceProc(VkDevice device, const char *name)
    {
        return reinterpret_cast<T>(getDeviceProcAddr(device, name));
    }

    void open()
    {
        library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!library)
            throw std::runtime_error(std::string("cannot load libvulkan.so.1: ") + dlerror());
        getInstanceProcAddr =
            reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
        requireProc(getInstanceProcAddr, "vkGetInstanceProcAddr");
        createInstance =
            requireProc(instanceProc<PFN_vkCreateInstance>(VK_NULL_HANDLE, "vkCreateInstance"),
                        "vkCreateInstance");
    }

    void loadInstance(VkInstance instance)
    {
        destroyInstance =
            requireProc(instanceProc<PFN_vkDestroyInstance>(instance, "vkDestroyInstance"),
                        "vkDestroyInstance");
        enumeratePhysicalDevices = requireProc(
            instanceProc<PFN_vkEnumeratePhysicalDevices>(instance, "vkEnumeratePhysicalDevices"),
            "vkEnumeratePhysicalDevices");
        getPhysicalDeviceQueueFamilyProperties =
            requireProc(instanceProc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
                            instance, "vkGetPhysicalDeviceQueueFamilyProperties"),
                        "vkGetPhysicalDeviceQueueFamilyProperties");
        getPhysicalDeviceProperties = requireProc(instanceProc<PFN_vkGetPhysicalDeviceProperties>(
                                                      instance, "vkGetPhysicalDeviceProperties"),
                                                  "vkGetPhysicalDeviceProperties");
        enumerateDeviceExtensionProperties =
            requireProc(instanceProc<PFN_vkEnumerateDeviceExtensionProperties>(
                            instance, "vkEnumerateDeviceExtensionProperties"),
                        "vkEnumerateDeviceExtensionProperties");
        getPhysicalDeviceFeatures2 = requireProc(instanceProc<PFN_vkGetPhysicalDeviceFeatures2>(
                                                     instance, "vkGetPhysicalDeviceFeatures2"),
                                                 "vkGetPhysicalDeviceFeatures2");
        getPhysicalDeviceMemoryProperties =
            requireProc(instanceProc<PFN_vkGetPhysicalDeviceMemoryProperties>(
                            instance, "vkGetPhysicalDeviceMemoryProperties"),
                        "vkGetPhysicalDeviceMemoryProperties");
        createDevice = requireProc(instanceProc<PFN_vkCreateDevice>(instance, "vkCreateDevice"),
                                   "vkCreateDevice");
        getDeviceProcAddr =
            requireProc(instanceProc<PFN_vkGetDeviceProcAddr>(instance, "vkGetDeviceProcAddr"),
                        "vkGetDeviceProcAddr");
    }

    void loadDevice(VkDevice device)
    {
#define LOAD_REQUIRED(member, type, name) member = requireProc(deviceProc<type>(device, name), name)
        LOAD_REQUIRED(destroyDevice, PFN_vkDestroyDevice, "vkDestroyDevice");
        LOAD_REQUIRED(getDeviceQueue, PFN_vkGetDeviceQueue, "vkGetDeviceQueue");
        LOAD_REQUIRED(deviceWaitIdle, PFN_vkDeviceWaitIdle, "vkDeviceWaitIdle");
        LOAD_REQUIRED(createTensor, PFN_vkCreateTensorARM, "vkCreateTensorARM");
        LOAD_REQUIRED(destroyTensor, PFN_vkDestroyTensorARM, "vkDestroyTensorARM");
        LOAD_REQUIRED(getTensorMemoryRequirements, PFN_vkGetTensorMemoryRequirementsARM,
                      "vkGetTensorMemoryRequirementsARM");
        LOAD_REQUIRED(bindTensorMemory, PFN_vkBindTensorMemoryARM, "vkBindTensorMemoryARM");
        LOAD_REQUIRED(createTensorView, PFN_vkCreateTensorViewARM, "vkCreateTensorViewARM");
        LOAD_REQUIRED(destroyTensorView, PFN_vkDestroyTensorViewARM, "vkDestroyTensorViewARM");
        LOAD_REQUIRED(allocateMemory, PFN_vkAllocateMemory, "vkAllocateMemory");
        LOAD_REQUIRED(freeMemory, PFN_vkFreeMemory, "vkFreeMemory");
        LOAD_REQUIRED(createBuffer, PFN_vkCreateBuffer, "vkCreateBuffer");
        LOAD_REQUIRED(destroyBuffer, PFN_vkDestroyBuffer, "vkDestroyBuffer");
        LOAD_REQUIRED(getBufferMemoryRequirements, PFN_vkGetBufferMemoryRequirements,
                      "vkGetBufferMemoryRequirements");
        LOAD_REQUIRED(bindBufferMemory, PFN_vkBindBufferMemory, "vkBindBufferMemory");
        LOAD_REQUIRED(mapMemory, PFN_vkMapMemory, "vkMapMemory");
        LOAD_REQUIRED(unmapMemory, PFN_vkUnmapMemory, "vkUnmapMemory");
        LOAD_REQUIRED(flushMappedMemoryRanges, PFN_vkFlushMappedMemoryRanges,
                      "vkFlushMappedMemoryRanges");
        LOAD_REQUIRED(createDescriptorSetLayout, PFN_vkCreateDescriptorSetLayout,
                      "vkCreateDescriptorSetLayout");
        LOAD_REQUIRED(destroyDescriptorSetLayout, PFN_vkDestroyDescriptorSetLayout,
                      "vkDestroyDescriptorSetLayout");
        LOAD_REQUIRED(createPipelineLayout, PFN_vkCreatePipelineLayout, "vkCreatePipelineLayout");
        LOAD_REQUIRED(destroyPipelineLayout, PFN_vkDestroyPipelineLayout,
                      "vkDestroyPipelineLayout");
        LOAD_REQUIRED(createShaderModule, PFN_vkCreateShaderModule, "vkCreateShaderModule");
        LOAD_REQUIRED(destroyShaderModule, PFN_vkDestroyShaderModule, "vkDestroyShaderModule");
        LOAD_REQUIRED(createDataGraphPipelines, PFN_vkCreateDataGraphPipelinesARM,
                      "vkCreateDataGraphPipelinesARM");
        LOAD_REQUIRED(destroyPipeline, PFN_vkDestroyPipeline, "vkDestroyPipeline");
        LOAD_REQUIRED(createDataGraphPipelineSession, PFN_vkCreateDataGraphPipelineSessionARM,
                      "vkCreateDataGraphPipelineSessionARM");
        LOAD_REQUIRED(destroyDataGraphPipelineSession, PFN_vkDestroyDataGraphPipelineSessionARM,
                      "vkDestroyDataGraphPipelineSessionARM");
        LOAD_REQUIRED(getSessionBindPointRequirements,
                      PFN_vkGetDataGraphPipelineSessionBindPointRequirementsARM,
                      "vkGetDataGraphPipelineSessionBindPointRequirementsARM");
        LOAD_REQUIRED(getSessionMemoryRequirements,
                      PFN_vkGetDataGraphPipelineSessionMemoryRequirementsARM,
                      "vkGetDataGraphPipelineSessionMemoryRequirementsARM");
        LOAD_REQUIRED(bindSessionMemory, PFN_vkBindDataGraphPipelineSessionMemoryARM,
                      "vkBindDataGraphPipelineSessionMemoryARM");
        LOAD_REQUIRED(createDescriptorPool, PFN_vkCreateDescriptorPool, "vkCreateDescriptorPool");
        LOAD_REQUIRED(destroyDescriptorPool, PFN_vkDestroyDescriptorPool,
                      "vkDestroyDescriptorPool");
        LOAD_REQUIRED(allocateDescriptorSets, PFN_vkAllocateDescriptorSets,
                      "vkAllocateDescriptorSets");
        LOAD_REQUIRED(updateDescriptorSets, PFN_vkUpdateDescriptorSets, "vkUpdateDescriptorSets");
        LOAD_REQUIRED(createCommandPool, PFN_vkCreateCommandPool, "vkCreateCommandPool");
        LOAD_REQUIRED(destroyCommandPool, PFN_vkDestroyCommandPool, "vkDestroyCommandPool");
        LOAD_REQUIRED(allocateCommandBuffers, PFN_vkAllocateCommandBuffers,
                      "vkAllocateCommandBuffers");
        LOAD_REQUIRED(beginCommandBuffer, PFN_vkBeginCommandBuffer, "vkBeginCommandBuffer");
        LOAD_REQUIRED(endCommandBuffer, PFN_vkEndCommandBuffer, "vkEndCommandBuffer");
        LOAD_REQUIRED(cmdBindPipeline, PFN_vkCmdBindPipeline, "vkCmdBindPipeline");
        LOAD_REQUIRED(cmdBindDescriptorSets, PFN_vkCmdBindDescriptorSets,
                      "vkCmdBindDescriptorSets");
        LOAD_REQUIRED(cmdDispatchDataGraph, PFN_vkCmdDispatchDataGraphARM,
                      "vkCmdDispatchDataGraphARM");
        cmdBeginConditionalRendering = deviceProc<PFN_vkCmdBeginConditionalRenderingEXT>(
            device, "vkCmdBeginConditionalRenderingEXT");
        cmdEndConditionalRendering = deviceProc<PFN_vkCmdEndConditionalRenderingEXT>(
            device, "vkCmdEndConditionalRenderingEXT");
        LOAD_REQUIRED(cmdExecuteCommands, PFN_vkCmdExecuteCommands, "vkCmdExecuteCommands");
        LOAD_REQUIRED(cmdCopyTensor, PFN_vkCmdCopyTensorARM, "vkCmdCopyTensorARM");
        cmdPipelineBarrier2 =
            deviceProc<PFN_vkCmdPipelineBarrier2>(device, "vkCmdPipelineBarrier2");
        cmdPipelineBarrier2KHR =
            deviceProc<PFN_vkCmdPipelineBarrier2KHR>(device, "vkCmdPipelineBarrier2KHR");
        LOAD_REQUIRED(createFence, PFN_vkCreateFence, "vkCreateFence");
        LOAD_REQUIRED(destroyFence, PFN_vkDestroyFence, "vkDestroyFence");
        LOAD_REQUIRED(resetFences, PFN_vkResetFences, "vkResetFences");
        LOAD_REQUIRED(waitForFences, PFN_vkWaitForFences, "vkWaitForFences");
        LOAD_REQUIRED(getFenceStatus, PFN_vkGetFenceStatus, "vkGetFenceStatus");
        LOAD_REQUIRED(queueSubmit, PFN_vkQueueSubmit, "vkQueueSubmit");
        queueSubmit2 = deviceProc<PFN_vkQueueSubmit2>(device, "vkQueueSubmit2");
        queueSubmit2KHR = deviceProc<PFN_vkQueueSubmit2KHR>(device, "vkQueueSubmit2KHR");
        LOAD_REQUIRED(createSemaphore, PFN_vkCreateSemaphore, "vkCreateSemaphore");
        LOAD_REQUIRED(destroySemaphore, PFN_vkDestroySemaphore, "vkDestroySemaphore");
#undef LOAD_REQUIRED
    }
};

struct TensorResource
{
    VkTensorARM tensor{};
    VkTensorViewARM view{};
    VkDeviceMemory memory{};
};
struct GraphResource
{
    VkDescriptorSetLayout descriptorSetLayout{};
    VkPipelineLayout pipelineLayout{};
    VkDescriptorPool descriptorPool{};
    VkDescriptorSet descriptorSet{};
    VkPipeline pipeline{};
    VkDataGraphPipelineSessionARM session{};
    std::vector<VkDeviceMemory> sessionMemory;
};

} // namespace

std::size_t graphConstantByteSize(const GraphConstantSpec &spec)
{
    std::size_t elements = 1;
    for (std::uint32_t index = 0; index < spec.dimensionCount; ++index)
        elements *= static_cast<std::size_t>(spec.dimensions[index]);
    const std::size_t elementBytes = spec.format == VK_FORMAT_R32_SINT ? 4U : 1U;
    return elements * elementBytes;
}

struct VulkanApp::Impl
{
    explicit Impl(const Options &value) : options(value)
    {
        try
        {
            initialize();
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }
    ~Impl()
    {
        cleanup();
    }

    Options options;
    DispatchTable vk;
    VkInstance instance{};
    VkPhysicalDevice physicalDevice{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    std::uint32_t queueFamily{};
    std::uint32_t queueCount{};
    VkDevice device{};
    VkDevice secondDevice{};
    PFN_vkDeviceWaitIdle secondDeviceWaitIdle{};
    PFN_vkDestroyDevice secondDeviceDestroy{};
    std::array<VkQueue, 2> queues{};
    VkShaderModule shaderModule{};
    VkBuffer conditionalPredicateBuffer{};
    VkDeviceMemory conditionalPredicateMemory{};
    std::array<TensorResource, 7> tensors{};
    std::vector<GraphResource> graphs;
    VkCommandPool commandPool{};
    std::vector<VkFence> liveFences;
    std::vector<VkSemaphore> liveSemaphores;
    std::uint32_t queueCursor{};

    void initialize()
    {
        writeSpirvReference();
        vk.open();
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.pApplicationName = "Neural Statistics Vulkan Fixture";
        application.applicationVersion = 1;
        application.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        createInfo.pApplicationInfo = &application;
        check(vk.createInstance(&createInfo, nullptr, &instance), "vkCreateInstance");
        vk.loadInstance(instance);
        selectPhysicalDevice();
        createLogicalDevice();
        vk.loadDevice(device);
        if (secondDevice)
        {
            secondDeviceWaitIdle =
                requireProc(vk.deviceProc<PFN_vkDeviceWaitIdle>(secondDevice, "vkDeviceWaitIdle"),
                            "vkDeviceWaitIdle(second device)");
            secondDeviceDestroy =
                requireProc(vk.deviceProc<PFN_vkDestroyDevice>(secondDevice, "vkDestroyDevice"),
                            "vkDestroyDevice(second device)");
            check(secondDeviceWaitIdle(secondDevice), "vkDeviceWaitIdle(second device)");
            emitEvent("second_logical_device_used", 0, 0, VK_SUCCESS);
        }
        validateRequestedSubmitRoute();
        validateConditionalRendering();
        for (std::uint32_t index = 0; index < queueCount; ++index)
            vk.getDeviceQueue(device, queueFamily, index, &queues[index]);
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamily;
        check(vk.createCommandPool(device, &poolInfo, nullptr, &commandPool),
              "vkCreateCommandPool");
        liveFences.reserve(3);
        liveSemaphores.reserve(1);
        if (options.secondDispatchConditionalFalse)
            createConditionalPredicate();
        createShaderModule();
        createTensors();
        graphs.reserve(options.graphCount);
        for (std::uint32_t index = 0; index < options.graphCount; ++index)
            createGraph(index);
    }

    void validateRequestedSubmitRoute() const
    {
        if (options.submitRoute == "core" && !vk.queueSubmit2)
        {
            throw std::runtime_error(
                "requested core submit route is unavailable: vkQueueSubmit2 was not exposed");
        }
        if (options.submitRoute == "khr" && !vk.queueSubmit2KHR)
        {
            throw std::runtime_error(
                "requested KHR submit route is unavailable: vkQueueSubmit2KHR was not exposed");
        }
    }

    void validateConditionalRendering() const
    {
        if (options.secondDispatchConditionalFalse &&
            (!vk.cmdBeginConditionalRendering || !vk.cmdEndConditionalRendering))
        {
            throw std::runtime_error(
                "conditional-rendering commands were not exposed by the device");
        }
    }

    void writeSpirvReference()
    {
        if (options.spirvReferencePath.empty())
            return;
        const std::filesystem::path output(options.spirvReferencePath);
        if (output.has_parent_path())
            std::filesystem::create_directories(output.parent_path());
        std::ofstream stream(output, std::ios::binary | std::ios::trunc);
        if (!stream)
            throw std::runtime_error("cannot create fixture SPIR-V reference " + output.string());
        stream.write(reinterpret_cast<const char *>(fixture_graph_spirv),
                     static_cast<std::streamsize>(fixture_graph_spirv_size));
        if (!stream)
            throw std::runtime_error("cannot write fixture SPIR-V reference " + output.string());
        emitEvent("spirv_reference_written", 0, 0, VK_SUCCESS, fixture_graph_spirv_size);
    }

    std::vector<VkExtensionProperties> deviceExtensions(VkPhysicalDevice candidate)
    {
        std::uint32_t count = 0;
        check(vk.enumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr),
              "vkEnumerateDeviceExtensionProperties(count)");
        std::vector<VkExtensionProperties> values(count);
        check(vk.enumerateDeviceExtensionProperties(candidate, nullptr, &count, values.data()),
              "vkEnumerateDeviceExtensionProperties");
        values.resize(count);
        return values;
    }

    bool supportsRequiredFeatures(VkPhysicalDevice candidate,
                                  const VkPhysicalDeviceProperties &properties) const
    {
        if (properties.apiVersion < VK_API_VERSION_1_3)
            return false;
        VkPhysicalDeviceConditionalRenderingFeaturesEXT availableConditional{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
        VkPhysicalDeviceVulkan12Features available12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan13Features available13{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceTensorFeaturesARM availableTensor{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM};
        VkPhysicalDeviceDataGraphFeaturesARM availableGraph{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM};
        availableGraph.pNext = &availableTensor;
        availableTensor.pNext = &available13;
        available13.pNext = &available12;
        availableConditional.pNext = &availableGraph;
        VkPhysicalDeviceFeatures2 available{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        available.pNext = options.secondDispatchConditionalFalse
                              ? static_cast<void *>(&availableConditional)
                              : static_cast<void *>(&availableGraph);
        vk.getPhysicalDeviceFeatures2(candidate, &available);
        return availableGraph.dataGraph && availableTensor.tensors &&
               availableTensor.tensorNonPacked && available12.storageBuffer8BitAccess &&
               available12.shaderInt8 && available13.synchronization2 &&
               available.features.shaderInt16 && available.features.shaderInt64 &&
               (!options.secondDispatchConditionalFalse ||
                availableConditional.conditionalRendering);
    }

    void selectPhysicalDevice()
    {
        std::uint32_t count = 0;
        check(vk.enumeratePhysicalDevices(instance, &count, nullptr),
              "vkEnumeratePhysicalDevices(count)");
        if (!count)
            throw std::runtime_error("no Vulkan physical device is available");
        std::vector<VkPhysicalDevice> devices(count);
        check(vk.enumeratePhysicalDevices(instance, &count, devices.data()),
              "vkEnumeratePhysicalDevices");
        int bestScore = -1;
        std::uint32_t bestFamily = 0;
        VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
        for (VkPhysicalDevice candidate : devices)
        {
            const auto extensions = deviceExtensions(candidate);
            if (!containsExtension(extensions, VK_ARM_DATA_GRAPH_EXTENSION_NAME) ||
                !containsExtension(extensions, VK_ARM_TENSORS_EXTENSION_NAME) ||
                (options.secondDispatchConditionalFalse &&
                 !containsExtension(extensions, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME)))
                continue;
            if (options.submitRoute == "khr" &&
                !containsExtension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME))
                continue;
            std::uint32_t familyCount = 0;
            vk.getPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
            std::vector<VkQueueFamilyProperties> families(familyCount);
            vk.getPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());
            VkPhysicalDeviceProperties properties{};
            vk.getPhysicalDeviceProperties(candidate, &properties);
            if (!supportsRequiredFeatures(candidate, properties))
                continue;
            int score = 0;
            switch (properties.deviceType)
            {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                score = 5;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                score = 4;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                score = 3;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                score = 2;
                break;
            default:
                score = 1;
                break;
            }
            for (std::uint32_t family = 0; family < familyCount; ++family)
            {
                const VkQueueFlags flags = families[family].queueFlags;
                const bool supportsConditionalCommands =
                    !options.secondDispatchConditionalFalse ||
                    (flags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT));
                if ((flags & VK_QUEUE_DATA_GRAPH_BIT_ARM) && supportsConditionalCommands &&
                    families[family].queueCount >= options.queueCount && score > bestScore)
                {
                    bestScore = score;
                    bestDevice = candidate;
                    bestFamily = family;
                }
            }
        }
        if (bestDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("no physical device satisfies the required data-graph queue, "
                                     "conditional-rendering topology, extensions, "
                                     "graph/tensor/integer features, and synchronization2 support");
        }
        physicalDevice = bestDevice;
        queueFamily = bestFamily;
        queueCount = options.queueCount;
        vk.getPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
        VkPhysicalDeviceProperties selectedProperties{};
        vk.getPhysicalDeviceProperties(physicalDevice, &selectedProperties);
        std::cout << "Fixture Vulkan device: " << selectedProperties.deviceName << " (type "
                  << selectedProperties.deviceType << ") queue family " << queueFamily << std::endl;
    }

    void createLogicalDevice()
    {
        const auto extensions = deviceExtensions(physicalDevice);
        std::vector<const char *> enabledExtensions{VK_ARM_DATA_GRAPH_EXTENSION_NAME,
                                                    VK_ARM_TENSORS_EXTENSION_NAME};
        if (options.secondDispatchConditionalFalse)
            enabledExtensions.push_back(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
        const bool hasBfloat16 =
            containsExtension(extensions, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
        const bool hasFloat8 = containsExtension(extensions, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        const bool hasReplicated =
            containsExtension(extensions, VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);
        if (hasBfloat16)
            enabledExtensions.push_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
        if (hasFloat8)
            enabledExtensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        if (hasReplicated)
            enabledExtensions.push_back(VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);
        if (options.submitRoute == "khr")
        {
            if (!containsExtension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME))
                throw std::runtime_error("KHR submit route requires VK_KHR_synchronization2");
            enabledExtensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
        }

        VkPhysicalDeviceConditionalRenderingFeaturesEXT availableConditional{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
        VkPhysicalDeviceShaderBfloat16FeaturesKHR availableBfloat{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR};
        VkPhysicalDeviceShaderFloat8FeaturesEXT availableFloat8{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
        VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT availableReplicated{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT};
        VkPhysicalDeviceVulkan11Features available11{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceVulkan12Features available12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan13Features available13{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceTensorFeaturesARM availableTensor{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM};
        VkPhysicalDeviceDataGraphFeaturesARM availableGraph{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM};
        availableGraph.pNext = &availableTensor;
        availableTensor.pNext = &available13;
        available13.pNext = &available12;
        available12.pNext = &available11;
        void **tail = &available11.pNext;
        if (hasBfloat16)
        {
            *tail = &availableBfloat;
            tail = &availableBfloat.pNext;
        }
        if (hasFloat8)
        {
            *tail = &availableFloat8;
            tail = &availableFloat8.pNext;
        }
        if (hasReplicated)
        {
            *tail = &availableReplicated;
            tail = &availableReplicated.pNext;
        }
        *tail = nullptr;
        availableConditional.pNext = &availableGraph;
        VkPhysicalDeviceFeatures2 available{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        available.pNext = options.secondDispatchConditionalFalse
                              ? static_cast<void *>(&availableConditional)
                              : static_cast<void *>(&availableGraph);
        vk.getPhysicalDeviceFeatures2(physicalDevice, &available);
        if (!availableGraph.dataGraph || !availableTensor.tensors ||
            !availableTensor.tensorNonPacked || !available12.storageBuffer8BitAccess ||
            !available12.shaderInt8 || !available.features.shaderInt16 ||
            !available.features.shaderInt64)
        {
            throw std::runtime_error(
                "physical device is missing required graph/tensor/integer features");
        }
        if ((options.submitRoute != "legacy" || options.distinctPrimaryStageSemaphore) &&
            !available13.synchronization2)
        {
            throw std::runtime_error("submit2 topology requires synchronization2");
        }
        if (options.secondDispatchConditionalFalse && !availableConditional.conditionalRendering)
        {
            throw std::runtime_error(
                "conditional-false dispatch requires conditionalRendering support");
        }

        VkPhysicalDeviceConditionalRenderingFeaturesEXT requestedConditional{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
        requestedConditional.conditionalRendering =
            options.secondDispatchConditionalFalse ? VK_TRUE : VK_FALSE;
        VkPhysicalDeviceShaderBfloat16FeaturesKHR requestedBfloat{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR};
        requestedBfloat.shaderBFloat16Type = availableBfloat.shaderBFloat16Type;
        VkPhysicalDeviceShaderFloat8FeaturesEXT requestedFloat8{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
        requestedFloat8.shaderFloat8 = availableFloat8.shaderFloat8;
        VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT requestedReplicated{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT};
        requestedReplicated.shaderReplicatedComposites =
            availableReplicated.shaderReplicatedComposites;
        VkPhysicalDeviceVulkan11Features requested11{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        requested11.storageBuffer16BitAccess = available11.storageBuffer16BitAccess;
        requested11.uniformAndStorageBuffer16BitAccess =
            available11.uniformAndStorageBuffer16BitAccess;
        requested11.storagePushConstant16 = available11.storagePushConstant16;
        VkPhysicalDeviceVulkan12Features requested12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        requested12.storageBuffer8BitAccess = VK_TRUE;
        requested12.uniformAndStorageBuffer8BitAccess =
            available12.uniformAndStorageBuffer8BitAccess;
        requested12.shaderInt8 = VK_TRUE;
        requested12.shaderFloat16 = available12.shaderFloat16;
        requested12.vulkanMemoryModel = available12.vulkanMemoryModel;
        requested12.vulkanMemoryModelDeviceScope = available12.vulkanMemoryModelDeviceScope;
        requested12.hostQueryReset = available12.hostQueryReset;
        VkPhysicalDeviceVulkan13Features requested13{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        requested13.synchronization2 = available13.synchronization2;
        requested13.maintenance4 = available13.maintenance4;
        requested13.pipelineCreationCacheControl = available13.pipelineCreationCacheControl;
        VkPhysicalDeviceTensorFeaturesARM requestedTensor{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM};
        requestedTensor.tensorNonPacked = VK_TRUE;
        requestedTensor.shaderTensorAccess = availableTensor.shaderTensorAccess;
        requestedTensor.tensors = VK_TRUE;
        VkPhysicalDeviceDataGraphFeaturesARM requestedGraph{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM};
        requestedGraph.dataGraph = VK_TRUE;
        requestedGraph.pNext = options.secondDispatchConditionalFalse
                                   ? static_cast<void *>(&requestedConditional)
                                   : static_cast<void *>(&requestedTensor);
        requestedConditional.pNext = &requestedTensor;
        requestedTensor.pNext = &requested13;
        requested13.pNext = &requested12;
        requested12.pNext = &requested11;
        tail = &requested11.pNext;
        if (hasBfloat16)
        {
            *tail = &requestedBfloat;
            tail = &requestedBfloat.pNext;
        }
        if (hasFloat8)
        {
            *tail = &requestedFloat8;
            tail = &requestedFloat8.pNext;
        }
        if (hasReplicated)
        {
            *tail = &requestedReplicated;
            tail = &requestedReplicated.pNext;
        }
        *tail = nullptr;

        std::array<float, 2> priorities{1.0F, 1.0F};
        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = queueFamily;
        queueInfo.queueCount = queueCount;
        queueInfo.pQueuePriorities = priorities.data();
        VkPhysicalDeviceFeatures baseFeatures{};
        baseFeatures.shaderInt16 = VK_TRUE;
        baseFeatures.shaderInt64 = VK_TRUE;
        VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        createInfo.pNext = &requestedGraph;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueInfo;
        createInfo.enabledExtensionCount = static_cast<std::uint32_t>(enabledExtensions.size());
        createInfo.ppEnabledExtensionNames = enabledExtensions.data();
        createInfo.pEnabledFeatures = &baseFeatures;
        check(vk.createDevice(physicalDevice, &createInfo, nullptr, &device), "vkCreateDevice");
        if (options.deviceCount == 2)
        {
            check(vk.createDevice(physicalDevice, &createInfo, nullptr, &secondDevice),
                  "vkCreateDevice(second logical device)");
        }
    }

    std::optional<std::uint32_t> findMemoryType(std::uint32_t bits,
                                                VkMemoryPropertyFlags required) const
    {
        for (std::uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index)
        {
            if ((bits & (1U << index)) &&
                (memoryProperties.memoryTypes[index].propertyFlags & required) == required)
                return index;
        }
        return std::nullopt;
    }

    std::uint32_t memoryType(std::uint32_t bits, VkMemoryPropertyFlags preferred) const
    {
        if (const auto preferredType = findMemoryType(bits, preferred))
            return *preferredType;
        if (const auto compatibleType = findMemoryType(bits, 0))
            return *compatibleType;
        throw std::runtime_error("no compatible Vulkan memory type");
    }

    VkDeviceMemory allocate(const VkMemoryRequirements &requirements, std::uint32_t memoryTypeIndex)
    {
        VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        info.allocationSize = requirements.size;
        info.memoryTypeIndex = memoryTypeIndex;
        VkDeviceMemory memory{};
        check(vk.allocateMemory(device, &info, nullptr, &memory), "vkAllocateMemory");
        return memory;
    }

    VkDeviceMemory allocate(const VkMemoryRequirements &requirements)
    {
        return allocate(requirements, memoryType(requirements.memoryTypeBits,
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    }

    void zeroHostVisibleMemory(VkDeviceMemory memory, VkDeviceSize size,
                               std::uint32_t memoryTypeIndex)
    {
        void *mapped = nullptr;
        check(vk.mapMemory(device, memory, 0, size, 0, &mapped), "vkMapMemory(input tensor)");
        std::memset(mapped, 0, static_cast<std::size_t>(size));
        const auto properties = memoryProperties.memoryTypes[memoryTypeIndex].propertyFlags;
        if (!(properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = memory;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            try
            {
                check(vk.flushMappedMemoryRanges(device, 1, &range),
                      "vkFlushMappedMemoryRanges(input tensor)");
            }
            catch (...)
            {
                vk.unmapMemory(device, memory);
                throw;
            }
        }
        vk.unmapMemory(device, memory);
    }

    void createConditionalPredicate()
    {
        VkBufferCreateInfo createInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        createInfo.size = sizeof(std::uint32_t);
        createInfo.usage = VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vk.createBuffer(device, &createInfo, nullptr, &conditionalPredicateBuffer),
              "vkCreateBuffer(conditional predicate)");
        VkMemoryRequirements requirements{};
        vk.getBufferMemoryRequirements(device, conditionalPredicateBuffer, &requirements);
        const auto hostType =
            findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (!hostType)
            throw std::runtime_error("conditional predicate has no host-visible memory type");
        conditionalPredicateMemory = allocate(requirements, *hostType);
        check(
            vk.bindBufferMemory(device, conditionalPredicateBuffer, conditionalPredicateMemory, 0),
            "vkBindBufferMemory(conditional predicate)");
        zeroHostVisibleMemory(conditionalPredicateMemory, sizeof(std::uint32_t), *hostType);
    }

    void destroyConditionalPredicate() noexcept
    {
        if (conditionalPredicateBuffer)
            vk.destroyBuffer(device, conditionalPredicateBuffer, nullptr);
        if (conditionalPredicateMemory)
            vk.freeMemory(device, conditionalPredicateMemory, nullptr);
        conditionalPredicateBuffer = VK_NULL_HANDLE;
        conditionalPredicateMemory = VK_NULL_HANDLE;
    }

    void destroyTensorResource(TensorResource &tensor) noexcept
    {
        if (tensor.view)
            vk.destroyTensorView(device, tensor.view, nullptr);
        if (tensor.tensor)
            vk.destroyTensor(device, tensor.tensor, nullptr);
        if (tensor.memory)
            vk.freeMemory(device, tensor.memory, nullptr);
        tensor = {};
    }

    VkMemoryRequirements createAndBindTensor(
        TensorResource &tensor, const std::array<std::int64_t, 4> &shape,
        VkTensorUsageFlagsARM usage, std::optional<std::uint32_t> forcedMemoryType = std::nullopt)
    {
        VkTensorDescriptionARM description{VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM};
        description.tiling = VK_TENSOR_TILING_LINEAR_ARM;
        description.format = tensorFormat;
        description.dimensionCount = static_cast<std::uint32_t>(shape.size());
        description.pDimensions = shape.data();
        description.usage = usage;
        VkTensorCreateInfoARM createInfo{VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM};
        createInfo.pDescription = &description;
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vk.createTensor(device, &createInfo, nullptr, &tensor.tensor), "vkCreateTensorARM");
        VkTensorMemoryRequirementsInfoARM requirementsInfo{
            VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM};
        requirementsInfo.tensor = tensor.tensor;
        VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        vk.getTensorMemoryRequirements(device, &requirementsInfo, &requirements);
        const auto type = forcedMemoryType.value_or(memoryType(
            requirements.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
        tensor.memory = allocate(requirements.memoryRequirements, type);
        VkBindTensorMemoryInfoARM bindInfo{VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM};
        bindInfo.tensor = tensor.tensor;
        bindInfo.memory = tensor.memory;
        check(vk.bindTensorMemory(device, 1, &bindInfo), "vkBindTensorMemoryARM");
        return requirements.memoryRequirements;
    }

    void initializeInputWithStaging()
    {
        TensorResource staging;
        bool submitted = false;
        try
        {
            if (!vk.cmdPipelineBarrier2 && !vk.cmdPipelineBarrier2KHR)
            {
                throw std::runtime_error("input tensor staging requires vkCmdPipelineBarrier2 or "
                                         "vkCmdPipelineBarrier2KHR");
            }
            VkTensorDescriptionARM description{VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM};
            description.tiling = VK_TENSOR_TILING_LINEAR_ARM;
            description.format = tensorFormat;
            description.dimensionCount = static_cast<std::uint32_t>(inputShape.size());
            description.pDimensions = inputShape.data();
            description.usage = VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM;
            VkTensorCreateInfoARM createInfo{VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM};
            createInfo.pDescription = &description;
            createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            check(vk.createTensor(device, &createInfo, nullptr, &staging.tensor),
                  "vkCreateTensorARM(input staging)");
            VkTensorMemoryRequirementsInfoARM requirementsInfo{
                VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM};
            requirementsInfo.tensor = staging.tensor;
            VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
            vk.getTensorMemoryRequirements(device, &requirementsInfo, &requirements);
            const auto hostType = findMemoryType(requirements.memoryRequirements.memoryTypeBits,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            if (!hostType)
                throw std::runtime_error("input tensor requires staging, but no host-visible "
                                         "staging tensor memory type is available");
            staging.memory = allocate(requirements.memoryRequirements, *hostType);
            VkBindTensorMemoryInfoARM bindInfo{VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM};
            bindInfo.tensor = staging.tensor;
            bindInfo.memory = staging.memory;
            check(vk.bindTensorMemory(device, 1, &bindInfo),
                  "vkBindTensorMemoryARM(input staging)");
            zeroHostVisibleMemory(staging.memory, requirements.memoryRequirements.size, *hostType);

            VkCommandBuffer commandBuffer = allocateCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vk.beginCommandBuffer(commandBuffer, &begin),
                  "vkBeginCommandBuffer(input initialization)");
            std::array<std::uint64_t, 4> offsets{};
            std::array<std::uint64_t, 4> extent{};
            std::transform(inputShape.begin(), inputShape.end(), extent.begin(),
                           [](std::int64_t value) { return static_cast<std::uint64_t>(value); });
            VkTensorCopyARM region{VK_STRUCTURE_TYPE_TENSOR_COPY_ARM};
            region.dimensionCount = static_cast<std::uint32_t>(extent.size());
            region.pSrcOffset = offsets.data();
            region.pDstOffset = offsets.data();
            region.pExtent = extent.data();
            VkCopyTensorInfoARM copy{VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_ARM};
            copy.srcTensor = staging.tensor;
            copy.dstTensor = tensors[0].tensor;
            copy.regionCount = 1;
            copy.pRegions = &region;
            vk.cmdCopyTensor(commandBuffer, &copy);
            VkTensorMemoryBarrierARM barrier{VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_ARM};
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM;
            barrier.dstAccessMask = VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.tensor = tensors[0].tensor;
            VkTensorDependencyInfoARM tensorDependency{
                VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_ARM};
            tensorDependency.tensorMemoryBarrierCount = 1;
            tensorDependency.pTensorMemoryBarriers = &barrier;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.pNext = &tensorDependency;
            if (vk.cmdPipelineBarrier2)
                vk.cmdPipelineBarrier2(commandBuffer, &dependency);
            else
                vk.cmdPipelineBarrier2KHR(commandBuffer, &dependency);
            check(vk.endCommandBuffer(commandBuffer), "vkEndCommandBuffer(input initialization)");
            VkFence fence = makeFence();
            VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            check(vk.queueSubmit(queues[0], 1, &submitInfo, fence),
                  "vkQueueSubmit(input initialization)");
            submitted = true;
            waitFence(fence);
            destroyFence(fence);
            emitEvent("input_tensor_zero_initialized", 0, 0, 1);
        }
        catch (...)
        {
            if (submitted)
                vk.deviceWaitIdle(device);
            destroyTensorResource(staging);
            throw;
        }
        destroyTensorResource(staging);
    }

    void createShaderModule()
    {
        if (fixture_graph_spirv_size % sizeof(std::uint32_t) != 0)
            throw std::runtime_error("embedded SPIR-V size is not word aligned");
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = fixture_graph_spirv_size;
        info.pCode = reinterpret_cast<const std::uint32_t *>(fixture_graph_spirv);
        check(vk.createShaderModule(device, &info, nullptr, &shaderModule), "vkCreateShaderModule");
    }

    void createTensors()
    {
        bool inputNeedsStaging = false;
        for (std::uint32_t index = 0; index < tensors.size(); ++index)
        {
            const auto &shape = index == 0 ? inputShape : outputShape;
            const auto usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM | VK_TENSOR_USAGE_SHADER_BIT_ARM |
                               VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM |
                               VK_TENSOR_USAGE_TRANSFER_DST_BIT_ARM;
            std::optional<std::uint32_t> hostType;
            VkMemoryRequirements requirements{};
            if (index == 0)
            {
                VkTensorDescriptionARM description{VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM};
                description.tiling = VK_TENSOR_TILING_LINEAR_ARM;
                description.format = tensorFormat;
                description.dimensionCount = static_cast<std::uint32_t>(shape.size());
                description.pDimensions = shape.data();
                description.usage = usage;
                VkTensorCreateInfoARM createInfo{VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM};
                createInfo.pDescription = &description;
                createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                check(vk.createTensor(device, &createInfo, nullptr, &tensors[index].tensor),
                      "vkCreateTensorARM");
                VkTensorMemoryRequirementsInfoARM requirementsInfo{
                    VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM};
                requirementsInfo.tensor = tensors[index].tensor;
                VkMemoryRequirements2 requirements2{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
                vk.getTensorMemoryRequirements(device, &requirementsInfo, &requirements2);
                requirements = requirements2.memoryRequirements;
                hostType = findMemoryType(requirements.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                const auto type = hostType.value_or(
                    memoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
                tensors[index].memory = allocate(requirements, type);
                VkBindTensorMemoryInfoARM bindInfo{VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM};
                bindInfo.tensor = tensors[index].tensor;
                bindInfo.memory = tensors[index].memory;
                check(vk.bindTensorMemory(device, 1, &bindInfo), "vkBindTensorMemoryARM");
            }
            else
            {
                requirements = createAndBindTensor(tensors[index], shape, usage);
            }
            VkTensorViewCreateInfoARM viewInfo{VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM};
            viewInfo.tensor = tensors[index].tensor;
            viewInfo.format = tensorFormat;
            check(vk.createTensorView(device, &viewInfo, nullptr, &tensors[index].view),
                  "vkCreateTensorViewARM");
            if (index == 0)
            {
                if (hostType)
                {
                    zeroHostVisibleMemory(tensors[index].memory, requirements.size, *hostType);
                    emitEvent("input_tensor_zero_initialized", 0, 0, 0);
                }
                else
                {
                    inputNeedsStaging = true;
                }
            }
        }
        if (inputNeedsStaging)
            initializeInputWithStaging();
    }

    void createGraph(std::uint32_t graphIndex)
    {
        graphs.emplace_back();
        GraphResource &graph = graphs.back();
        try
        {
            std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
            for (std::uint32_t binding = 0; binding < bindings.size(); ++binding)
            {
                bindings[binding].binding = binding;
                bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
                bindings[binding].descriptorCount = 1;
                bindings[binding].stageFlags = VK_SHADER_STAGE_ALL;
            }
            VkDescriptorSetLayoutCreateInfo setInfo{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            setInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
            setInfo.pBindings = bindings.data();
            check(
                vk.createDescriptorSetLayout(device, &setInfo, nullptr, &graph.descriptorSetLayout),
                "vkCreateDescriptorSetLayout");
            VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layoutInfo.setLayoutCount = 1;
            layoutInfo.pSetLayouts = &graph.descriptorSetLayout;
            check(vk.createPipelineLayout(device, &layoutInfo, nullptr, &graph.pipelineLayout),
                  "vkCreatePipelineLayout");

            std::array<VkTensorDescriptionARM, 7> resourceDescriptions{};
            std::array<VkDataGraphPipelineResourceInfoARM, 7> resourceInfos{};
            for (std::uint32_t binding = 0; binding < resourceInfos.size(); ++binding)
            {
                const auto &shape = binding == 0 ? inputShape : outputShape;
                auto &description = resourceDescriptions[binding];
                description.sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM;
                description.tiling = VK_TENSOR_TILING_LINEAR_ARM;
                description.format = tensorFormat;
                description.dimensionCount = static_cast<std::uint32_t>(shape.size());
                description.pDimensions = shape.data();
                description.usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
                auto &resource = resourceInfos[binding];
                resource.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM;
                resource.pNext = &description;
                resource.descriptorSet = 0;
                resource.binding = binding;
            }
            std::array<VkTensorDescriptionARM, graphConstantSpecs.size()> constantDescriptions{};
            std::array<VkDataGraphPipelineConstantARM, graphConstantSpecs.size()> constants{};
            std::size_t offset = 0;
            for (std::uint32_t index = 0; index < constants.size(); ++index)
            {
                const auto &spec = graphConstantSpecs[index];
                auto &description = constantDescriptions[index];
                description.sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM;
                description.tiling = VK_TENSOR_TILING_LINEAR_ARM;
                description.format = spec.format;
                description.dimensionCount = spec.dimensionCount;
                description.pDimensions = spec.dimensions.data();
                description.usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
                auto &constant = constants[index];
                constant.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM;
                constant.pNext = &description;
                constant.id = index;
                constant.pConstantData = fixture_graph_constants + offset;
                offset += graphConstantByteSize(spec);
            }
            if (offset != fixture_graph_constants_size)
                throw std::runtime_error(
                    "embedded graph constant metadata does not match payload size");
            VkDataGraphPipelineNeuralStatisticsCreateInfoARM appStatistics{
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_NEURAL_STATISTICS_CREATE_INFO_ARM, nullptr,
                VK_TRUE};
            VkDataGraphPipelineShaderModuleCreateInfoARM shaderInfo{
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM};
            shaderInfo.module = shaderModule;
            shaderInfo.pName = graphEntryPoint;
            shaderInfo.constantCount = static_cast<std::uint32_t>(constants.size());
            shaderInfo.pConstants = constants.data();
            VkDataGraphPipelineCreateInfoARM pipelineInfo{
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM};
            appStatistics.pNext = &shaderInfo;
            pipelineInfo.pNext = options.noncapturable ? static_cast<const void *>(&appStatistics)
                                                       : static_cast<const void *>(&shaderInfo);
            pipelineInfo.layout = graph.pipelineLayout;
            pipelineInfo.resourceInfoCount = static_cast<std::uint32_t>(resourceInfos.size());
            pipelineInfo.pResourceInfos = resourceInfos.data();
            if (options.noncapturable && graphIndex == 0)
                emitEvent("noncapturable_state_enabled", 0, 0, 1);
            check(vk.createDataGraphPipelines(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1,
                                              &pipelineInfo, nullptr, &graph.pipeline),
                  "vkCreateDataGraphPipelinesARM");

            VkDataGraphPipelineSessionCreateInfoARM sessionInfo{
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM};
            sessionInfo.dataGraphPipeline = graph.pipeline;
            check(vk.createDataGraphPipelineSession(device, &sessionInfo, nullptr, &graph.session),
                  "vkCreateDataGraphPipelineSessionARM");
            bindSession(graph);

            VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_TENSOR_ARM, 7};
            VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            check(vk.createDescriptorPool(device, &poolInfo, nullptr, &graph.descriptorPool),
                  "vkCreateDescriptorPool");
            VkDescriptorSetAllocateInfo allocateInfo{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocateInfo.descriptorPool = graph.descriptorPool;
            allocateInfo.descriptorSetCount = 1;
            allocateInfo.pSetLayouts = &graph.descriptorSetLayout;
            check(vk.allocateDescriptorSets(device, &allocateInfo, &graph.descriptorSet),
                  "vkAllocateDescriptorSets");
            std::array<VkWriteDescriptorSetTensorARM, 7> tensorWrites{};
            std::array<VkWriteDescriptorSet, 7> writes{};
            for (std::uint32_t binding = 0; binding < writes.size(); ++binding)
            {
                tensorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM;
                tensorWrites[binding].tensorViewCount = 1;
                tensorWrites[binding].pTensorViews = &tensors[binding].view;
                writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[binding].pNext = &tensorWrites[binding];
                writes[binding].dstSet = graph.descriptorSet;
                writes[binding].dstBinding = binding;
                writes[binding].descriptorCount = 1;
                writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
            }
            vk.updateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()),
                                    writes.data(), 0, nullptr);
        }
        catch (...)
        {
            destroyGraph(graph);
            graphs.pop_back();
            throw;
        }
    }

    void bindSession(GraphResource &graph)
    {
        VkDataGraphPipelineSessionBindPointRequirementsInfoARM info{
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM};
        info.session = graph.session;
        std::uint32_t count = 0;
        check(vk.getSessionBindPointRequirements(device, &info, &count, nullptr),
              "vkGetDataGraphPipelineSessionBindPointRequirementsARM(count)");
        std::vector<VkDataGraphPipelineSessionBindPointRequirementARM> requirements(count);
        for (auto &requirement : requirements)
            requirement.sType =
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM;
        check(vk.getSessionBindPointRequirements(device, &info, &count, requirements.data()),
              "vkGetDataGraphPipelineSessionBindPointRequirementsARM");
        requirements.resize(count);
        std::vector<VkBindDataGraphPipelineSessionMemoryInfoARM> binds;
        for (const auto &requirement : requirements)
        {
            if (requirement.bindPointType !=
                VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM)
                continue;
            for (std::uint32_t objectIndex = 0; objectIndex < requirement.numObjects; ++objectIndex)
            {
                VkDataGraphPipelineSessionMemoryRequirementsInfoARM memoryInfo{
                    VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM};
                memoryInfo.session = graph.session;
                memoryInfo.bindPoint = requirement.bindPoint;
                memoryInfo.objectIndex = objectIndex;
                VkMemoryRequirements2 memoryRequirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
                vk.getSessionMemoryRequirements(device, &memoryInfo, &memoryRequirements);
                if (!memoryRequirements.memoryRequirements.size)
                    continue;
                graph.sessionMemory.push_back(allocate(memoryRequirements.memoryRequirements));
                VkBindDataGraphPipelineSessionMemoryInfoARM bind{
                    VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM};
                bind.session = graph.session;
                bind.bindPoint = requirement.bindPoint;
                bind.objectIndex = objectIndex;
                bind.memory = graph.sessionMemory.back();
                binds.push_back(bind);
            }
        }
        if (!binds.empty())
            check(vk.bindSessionMemory(device, static_cast<std::uint32_t>(binds.size()),
                                       binds.data()),
                  "vkBindDataGraphPipelineSessionMemoryARM");
    }

    VkCommandBuffer allocateCommandBuffer(VkCommandBufferLevel level)
    {
        VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        info.commandPool = commandPool;
        info.level = level;
        info.commandBufferCount = 1;
        VkCommandBuffer commandBuffer{};
        check(vk.allocateCommandBuffers(device, &info, &commandBuffer), "vkAllocateCommandBuffers");
        return commandBuffer;
    }

    VkCommandBuffer recordGraphs(std::uint32_t first, std::uint32_t count,
                                 VkCommandBufferLevel level, RecordingUse use,
                                 bool conditionFalse = false)
    {
        VkCommandBuffer commandBuffer = allocateCommandBuffer(level);
        VkCommandBufferInheritanceInfo inheritance{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = commandBufferUsage(use);
        if (level == VK_COMMAND_BUFFER_LEVEL_SECONDARY)
            begin.pInheritanceInfo = &inheritance;
        check(vk.beginCommandBuffer(commandBuffer, &begin), "vkBeginCommandBuffer");
        VkDataGraphPipelineDispatchInfoARM dispatchInfo{
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_DISPATCH_INFO_ARM};
        for (std::uint32_t index = first; index < first + count; ++index)
        {
            auto &graph = graphs[index];
            vk.cmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM,
                               graph.pipeline);
            vk.cmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM,
                                     graph.pipelineLayout, 0, 1, &graph.descriptorSet, 0, nullptr);
            if (conditionFalse)
            {
                VkConditionalRenderingBeginInfoEXT conditional{
                    VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT};
                conditional.buffer = conditionalPredicateBuffer;
                vk.cmdBeginConditionalRendering(commandBuffer, &conditional);
            }
            vk.cmdDispatchDataGraph(commandBuffer, graph.session, &dispatchInfo);
            if (conditionFalse)
                vk.cmdEndConditionalRendering(commandBuffer);
        }
        check(vk.endCommandBuffer(commandBuffer), "vkEndCommandBuffer");
        return commandBuffer;
    }

    VkCommandBuffer recordSecondaryWrapper(VkCommandBuffer secondary, std::uint32_t executions,
                                           RecordingUse use)
    {
        VkCommandBuffer primary = allocateCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = commandBufferUsage(use);
        check(vk.beginCommandBuffer(primary, &begin), "vkBeginCommandBuffer(primary wrapper)");
        std::vector<VkCommandBuffer> repeated(executions, secondary);
        vk.cmdExecuteCommands(primary, static_cast<std::uint32_t>(repeated.size()),
                              repeated.data());
        check(vk.endCommandBuffer(primary), "vkEndCommandBuffer(primary wrapper)");
        emitSecondaryWrapperRecorded(executions);
        return primary;
    }

    VkFence makeFence()
    {
        VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence{};
        check(vk.createFence(device, &info, nullptr, &fence), "vkCreateFence");
        try
        {
            liveFences.push_back(fence);
        }
        catch (...)
        {
            vk.destroyFence(device, fence, nullptr);
            throw;
        }
        return fence;
    }

    void destroyFence(VkFence fence) noexcept
    {
        if (!fence)
            return;
        vk.destroyFence(device, fence, nullptr);
        const auto found = std::find(liveFences.begin(), liveFences.end(), fence);
        if (found != liveFences.end())
            liveFences.erase(found);
    }

    VkSemaphore makeSemaphore()
    {
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkSemaphore semaphore{};
        check(vk.createSemaphore(device, &info, nullptr, &semaphore), "vkCreateSemaphore");
        try
        {
            liveSemaphores.push_back(semaphore);
        }
        catch (...)
        {
            vk.destroySemaphore(device, semaphore, nullptr);
            throw;
        }
        return semaphore;
    }

    void destroySemaphore(VkSemaphore semaphore) noexcept
    {
        if (!semaphore)
            return;
        vk.destroySemaphore(device, semaphore, nullptr);
        const auto found = std::find(liveSemaphores.begin(), liveSemaphores.end(), semaphore);
        if (found != liveSemaphores.end())
            liveSemaphores.erase(found);
    }

    std::uint32_t selectQueue(std::uint32_t submission)
    {
        const std::uint32_t index = queueCount > 1 ? queueCursor++ % queueCount : 0;
        emitEvent("queue_selected", submission, 0, static_cast<int>(index));
        return index;
    }

    VkResult submit(std::uint32_t queueIndex, const std::vector<VkCommandBuffer> &commandBuffers,
                    VkFence fence, VkSemaphore waitSemaphore = VK_NULL_HANDLE,
                    VkSemaphore signalSemaphore = VK_NULL_HANDLE)
    {
        if (options.submitRoute == "legacy")
        {
            VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            info.commandBufferCount = static_cast<std::uint32_t>(commandBuffers.size());
            info.pCommandBuffers = commandBuffers.data();
            emitEvent("submit_route_legacy");
            return vk.queueSubmit(queues[queueIndex], 1, &info, fence);
        }
        std::vector<VkCommandBufferSubmitInfo> commandInfos(commandBuffers.size());
        for (std::size_t index = 0; index < commandBuffers.size(); ++index)
        {
            commandInfos[index].sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
            commandInfos[index].commandBuffer = commandBuffers[index];
        }
        VkSemaphoreSubmitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        waitInfo.semaphore = waitSemaphore;
        waitInfo.stageMask = VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM;
        VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signalInfo.semaphore = signalSemaphore;
        signalInfo.stageMask = VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM;
        VkSubmitInfo2 info{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        info.commandBufferInfoCount = static_cast<std::uint32_t>(commandInfos.size());
        info.pCommandBufferInfos = commandInfos.data();
        if (waitSemaphore)
        {
            info.waitSemaphoreInfoCount = 1;
            info.pWaitSemaphoreInfos = &waitInfo;
        }
        if (signalSemaphore)
        {
            info.signalSemaphoreInfoCount = 1;
            info.pSignalSemaphoreInfos = &signalInfo;
        }
        if (options.submitRoute == "core")
        {
            emitEvent("submit_route_core");
            if (!vk.queueSubmit2)
                throw std::runtime_error("vkQueueSubmit2 is unavailable");
            return vk.queueSubmit2(queues[queueIndex], 1, &info, fence);
        }
        emitEvent("submit_route_khr");
        if (!vk.queueSubmit2KHR)
            throw std::runtime_error("vkQueueSubmit2KHR is unavailable");
        return vk.queueSubmit2KHR(queues[queueIndex], 1, &info, fence);
    }

    void waitFence(VkFence fence)
    {
        const std::uint64_t timeout =
            static_cast<std::uint64_t>(options.reuseTimeoutMs) * 1000ULL * 1000ULL;
        check(vk.waitForFences(device, 1, &fence, VK_TRUE, timeout), "vkWaitForFences");
    }

    void submitWithRetry(std::uint32_t queueIndex,
                         const std::vector<VkCommandBuffer> &commandBuffers, VkFence fence,
                         std::uint32_t submission)
    {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(options.reuseTimeoutMs);
        if (options.immediateResubmission)
            emitEvent("immediate_resubmission_submit", submission, 0, VK_SUCCESS);
        for (std::uint32_t attempt = 0;; ++attempt)
        {
            emitEvent("submit_start", submission, attempt, VK_SUCCESS);
            const VkResult result = submit(queueIndex, commandBuffers, fence);
            if (result == VK_SUCCESS)
            {
                emitEvent(attempt == 0 ? "submit_end" : "reuse_retry_success", submission, attempt,
                          result);
                return;
            }
            emitEvent("submit_end", submission, attempt, result);
            if (options.immediateResubmission || submission == 0 ||
                result != VK_ERROR_VALIDATION_FAILED_EXT ||
                std::chrono::steady_clock::now() >= deadline)
                throw VulkanError("queue submit", result);
            emitEvent("reuse_transient_not_forwarded", submission, attempt, result);
            std::this_thread::sleep_for(std::chrono::milliseconds(options.reuseRetryMs));
        }
    }

    void runOrdinary()
    {
        if (options.graphCount > 1)
        {
            for (std::uint32_t graph = 0; graph < options.graphCount; ++graph)
            {
                VkCommandBuffer commandBuffer =
                    recordGraphs(graph, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY, RecordingUse::oneShot);
                VkFence fence = makeFence();
                const std::uint32_t queueIndex = selectQueue(graph);
                submitWithRetry(queueIndex, {commandBuffer}, fence, graph);
                waitFence(fence);
                emitApplicationFenceComplete(graph);
                destroyFence(fence);
            }
            return;
        }
        const auto recordingPlan = ordinaryRecordingPlan(options);
        VkCommandBuffer commandBuffer{};
        if (options.secondaryExecutions)
        {
            VkCommandBuffer secondary =
                recordGraphs(0, 1, VK_COMMAND_BUFFER_LEVEL_SECONDARY, recordingPlan.secondary);
            commandBuffer = recordSecondaryWrapper(secondary, options.secondaryExecutions,
                                                   recordingPlan.primary);
        }
        else
        {
            commandBuffer =
                recordGraphs(0, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY, recordingPlan.primary);
        }
        const std::vector<VkCommandBuffer> submitted(options.primaryCommandBuffersPerSubmit,
                                                     commandBuffer);
        if (options.primaryCommandBuffersPerSubmit > 1)
        {
            emitEvent("primary_command_buffer_batch", 0, 0, static_cast<int>(submitted.size()));
            emitEvent("primary_command_buffer_batch_marshaled", 0, 0,
                      static_cast<int>(submitted.size()));
        }
        const std::uint32_t submissionCount =
            std::max(options.resubmissions, options.dispatchRepeats);
        VkFence fence = makeFence();
        for (std::uint32_t submission = 0; submission < submissionCount; ++submission)
        {
            if (submission)
                check(vk.resetFences(device, 1, &fence), "vkResetFences");
            const std::uint32_t queueIndex = selectQueue(submission);
            submitWithRetry(queueIndex, submitted, fence, submission);
            if (options.destroyAfterFence)
            {
                emitEvent("application_fence_prompt_wait_start", submission, 0, VK_NOT_READY);
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::milliseconds(options.reuseTimeoutMs);
                while (vk.getFenceStatus(device, fence) == VK_NOT_READY)
                {
                    if (std::chrono::steady_clock::now() >= deadline)
                        throw std::runtime_error("timed out polling pending-destruction fence");
                    std::this_thread::yield();
                }
                emitEvent("application_fence_prompt_complete", submission, 0, VK_SUCCESS);
            }
            else
            {
                waitFence(fence);
            }
            emitApplicationFenceComplete(submission);
        }
        if (options.destroyAfterFence)
        {
            const char *capture = std::getenv("VK_LAYER_CAPTURE_FOLDER");
            const char *mode = std::getenv("VK_LAYER_STATISTICS_MODE");
            if (!capture || !mode)
                throw std::runtime_error(
                    "pending-work destruction requires layer capture settings");
            const std::filesystem::path raw = std::filesystem::path(capture) / "pipeline_000000" /
                                              "session_000000" / "dispatch_000000" /
                                              (std::string("statistics_mode") + mode + ".bin");
            if (std::filesystem::exists(raw))
            {
                emitEvent("collector_output_already_published", 0, 0, VK_SUCCESS);
                throw std::runtime_error(
                    "collector output was already published before application teardown");
            }
            emitEvent("collector_output_pending_observed", 0, 0, VK_NOT_READY);
            destroyGraphs();
        }
        destroyFence(fence);
    }

    void runDistinctPrimaries()
    {
        std::array<VkCommandBuffer, 2> commandBuffers{
            recordGraphs(0, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY, RecordingUse::oneShot),
            recordGraphs(0, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY, RecordingUse::oneShot,
                         options.secondDispatchConditionalFalse)};
        if (options.secondDispatchConditionalFalse)
        {
            emitEvent("conditional_false_dispatch_recorded", 1, 0, VK_SUCCESS);
        }
        std::array<VkFence, 2> fences{makeFence(), makeFence()};
        for (std::uint32_t submission = 0; submission < 2; ++submission)
            emitEvent("distinct_primary_recorded", submission, 0, VK_SUCCESS);
        VkSemaphore semaphore = makeSemaphore();
        for (std::uint32_t submission = 0; submission < 2; ++submission)
        {
            const std::uint32_t queueIndex = selectQueue(submission);
            VkSemaphore wait = VK_NULL_HANDLE;
            VkSemaphore signal = VK_NULL_HANDLE;
            if (options.distinctPrimaryStageSemaphore && submission == 0)
            {
                signal = semaphore;
                emitEvent("data_graph_stage_semaphore_signal", submission, 0, VK_SUCCESS);
            }
            else if (options.distinctPrimaryStageSemaphore)
            {
                wait = semaphore;
                emitEvent("data_graph_stage_semaphore_wait", submission, 0, VK_SUCCESS);
            }
            check(
                submit(queueIndex, {commandBuffers[submission]}, fences[submission], wait, signal),
                "distinct primary submit");
            emitEvent("distinct_primary_submit_return", submission, 0, VK_SUCCESS);
            if (!options.distinctPrimaryStageSemaphore)
            {
                waitFence(fences[submission]);
                emitApplicationFenceComplete(submission);
            }
        }
        if (options.distinctPrimaryStageSemaphore)
        {
            for (std::uint32_t submission = 0; submission < 2; ++submission)
            {
                waitFence(fences[submission]);
                emitApplicationFenceComplete(submission);
            }
        }
        for (VkFence fence : fences)
            destroyFence(fence);
        destroySemaphore(semaphore);
    }

    void runCapacity()
    {
        std::array<VkCommandBuffer, 3> commandBuffers{
            recordGraphs(0, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY, RecordingUse::oneShot),
            recordGraphs(1, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY, RecordingUse::oneShot),
            recordGraphs(2, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY, RecordingUse::oneShot)};
        std::array<VkFence, 3> fences{makeFence(), makeFence(), makeFence()};
        for (std::uint32_t submission = 0; submission < 2; ++submission)
        {
            emitEvent("capacity_submit_start", submission, 0, VK_SUCCESS);
            const std::uint32_t queueIndex = selectQueue(submission);
            check(submit(queueIndex, {commandBuffers[submission]}, fences[submission]),
                  "capacity submit");
            emitEvent("capacity_submit_return", submission, 0, VK_SUCCESS);
        }
        std::atomic<bool> returned{false};
        std::atomic<bool> waitObserved{false};
        std::exception_ptr observerFailure;
        std::jthread observer(
            [&](std::stop_token stopToken) noexcept
            {
                try
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    if (!returned.load(std::memory_order_acquire) && !stopToken.stop_requested())
                    {
                        waitObserved.store(true, std::memory_order_release);
                        emitEvent("capacity_wait_observed", 2, 0, VK_NOT_READY);
                    }
                    while (!returned.load(std::memory_order_acquire) && !stopToken.stop_requested())
                    {
                        for (std::uint32_t prior = 0; prior < 2; ++prior)
                        {
                            if (vk.getFenceStatus(device, fences[prior]) == VK_SUCCESS)
                            {
                                emitEvent("capacity_prior_application_fence_complete", prior, 0,
                                          VK_SUCCESS);
                                return;
                            }
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
                catch (...)
                {
                    observerFailure = std::current_exception();
                }
            });
        emitEvent("capacity_submit_start", 2, 0, VK_SUCCESS);
        const std::uint32_t queueIndex = selectQueue(2);
        check(submit(queueIndex, {commandBuffers[2]}, fences[2]), "capacity submit C");
        returned.store(true, std::memory_order_release);
        emitEvent("capacity_submit_return", 2, 0, VK_SUCCESS);
        observer.join();
        if (observerFailure)
            std::rethrow_exception(observerFailure);
        if (!waitObserved.load(std::memory_order_acquire))
            throw std::runtime_error(
                "capacity submission returned before a full-capacity wait was observed");
        for (std::uint32_t submission = 0; submission < 3; ++submission)
        {
            waitFence(fences[submission]);
            emitApplicationFenceComplete(submission);
            destroyFence(fences[submission]);
        }
    }

    void run()
    {
        if (options.capacitySubmissions == 3)
            runCapacity();
        else if (options.distinctPrimarySubmissions == 2)
            runDistinctPrimaries();
        else
            runOrdinary();
    }

    void destroyGraph(GraphResource &graph) noexcept
    {
        if (graph.session)
            vk.destroyDataGraphPipelineSession(device, graph.session, nullptr);
        if (graph.pipeline)
            vk.destroyPipeline(device, graph.pipeline, nullptr);
        if (graph.descriptorPool)
            vk.destroyDescriptorPool(device, graph.descriptorPool, nullptr);
        if (graph.pipelineLayout)
            vk.destroyPipelineLayout(device, graph.pipelineLayout, nullptr);
        if (graph.descriptorSetLayout)
            vk.destroyDescriptorSetLayout(device, graph.descriptorSetLayout, nullptr);
        for (VkDeviceMemory memory : graph.sessionMemory)
            if (memory)
                vk.freeMemory(device, memory, nullptr);
        graph = {};
    }

    void destroyGraphs()
    {
        emitEvent("destruction_begin", 0, 0, VK_SUCCESS);
        for (auto &graph : graphs)
            destroyGraph(graph);
        graphs.clear();
        emitEvent("destruction_end", 0, 0, VK_SUCCESS);
    }

    void cleanup() noexcept
    {
        if (device && vk.deviceWaitIdle)
            vk.deviceWaitIdle(device);
        if (device && vk.destroyFence)
        {
            for (VkFence fence : liveFences)
                vk.destroyFence(device, fence, nullptr);
            liveFences.clear();
        }
        if (device && vk.destroySemaphore)
        {
            for (VkSemaphore semaphore : liveSemaphores)
                vk.destroySemaphore(device, semaphore, nullptr);
            liveSemaphores.clear();
        }
        if (device && commandPool && vk.destroyCommandPool)
            vk.destroyCommandPool(device, commandPool, nullptr);
        if (device)
            destroyConditionalPredicate();
        if (device)
            for (auto &graph : graphs)
                destroyGraph(graph);
        graphs.clear();
        if (device && shaderModule && vk.destroyShaderModule)
            vk.destroyShaderModule(device, shaderModule, nullptr);
        if (device)
        {
            for (auto &tensor : tensors)
                destroyTensorResource(tensor);
        }
        if (secondDevice && secondDeviceWaitIdle)
            secondDeviceWaitIdle(secondDevice);
        if (secondDevice && secondDeviceDestroy)
            secondDeviceDestroy(secondDevice, nullptr);
        secondDevice = VK_NULL_HANDLE;
        if (device && vk.destroyDevice)
            vk.destroyDevice(device, nullptr);
        if (instance && vk.destroyInstance)
            vk.destroyInstance(instance, nullptr);
    }
};

VulkanApp::VulkanApp(const Options &options) : impl_(std::make_unique<Impl>(options)) {}
VulkanApp::~VulkanApp() = default;
void VulkanApp::run()
{
    impl_->run();
}

} // namespace fixture
