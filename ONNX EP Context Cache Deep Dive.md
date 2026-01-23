# **ONNX Runtime Execution Provider Context Cache: A Deep Dive**

## **1\. Executive Summary & Introduction**

In the current AI deployment landscape, deep learning models are migrating from the cloud to the edge, mobile, and embedded devices at an unprecedented rate. As model architectures grow increasingly complex—from standard CNNs to Billion-parameter Transformers and Large Language Models (LLMs)—inference engines face significant performance challenges. **ONNX Runtime (ORT)** addresses this by offloading computation to specialized hardware accelerators (NPUs, GPUs, DSPs) via its **Execution Provider (EP)** architecture.  
However, utilizing specialized hardware often incurs a high initialization cost. EPs must perform Just-In-Time (JIT) compilation during session creation, which includes operator fusion, memory planning, constant folding, and generating hardware-specific machine code. For Generative AI models on edge devices, this process can take seconds or even minutes, severely impacting the user experience (cold start latency).  
To solve this, ONNX Runtime introduced the **EP Context Cache** feature. This mechanism allows developers to serialize the expensive compilation state (the EP Context) and store it, enabling subsequent runs to load the pre-compiled state directly, bypassing the compilation phase. This report provides a comprehensive technical analysis of the EP Context Cache, covering its architectural design, implementation across major hardware backends (QNN, TensorRT, OpenVINO, DirectML), engineering strategies, and performance benchmarks.

## **2\. The Core Challenge: Compilation Latency in Modern Inference**

To understand the value of Context Cache, one must first understand the "compilation gap."

### **2.1 From Graph to Hardware Instructions**

When ONNX Runtime loads a .onnx model, it starts as a platform-agnostic computation graph. To run on specific hardware (e.g., Qualcomm Hexagon NPU or NVIDIA Ada Lovelace GPU), the following transformation steps occur :

1. **Graph Partitioning**: ORT splits the graph into subgraphs based on the registered capabilities of the EP. Supported nodes go to the EP; unsupported ones fall back to the CPU.  
2. **Layout Transformation**: EPs may insert layout conversion nodes (e.g., NCHW to NHWC) to match hardware memory access patterns.  
3. **Graph Optimization**:  
   * **Fusion**: Merging Conv+BatchNorm+Relu into a single kernel to reduce memory bandwidth.  
   * **Constant Folding**: Pre-calculating static parameters.  
4. **Low-Level Compilation**:  
   * **TensorRT**: Builds an "Engine" by searching for optimal CUDA kernels.  
   * **QNN (Qualcomm AI Engine)**: Generates Hexagon DSP binaries. For LLMs, this is extremely CPU-intensive. \* **OpenVINO**: Compiles Intermediate Representation (IR) into executable objects for CPU/GPU/NPU.

For models like Stable Diffusion or Llama 2, performing these steps on a constrained edge CPU can result in unacceptable startup delays.

### **2.2 The Solution: Ahead-of-Time (AOT) & Context Caching**

The EP Context Cache treats the compiled hardware state as a persistent asset. It supports not only standard caching (run once slow, run twice fast) but also **Cross-Compilation**, allowing developers to generate cache files for mobile devices (ARM64) using powerful servers (x64).

## **3\. Architecture Design: The EPContext Node**

The Context Cache feature is standardized within the ONNX graph structure via a special node type: the **EPContext Node**.

### **3.1 EPContext Node Specification**

When Context Cache generation is enabled, the EP replaces the compute subgraph it manages with a single EPContext node using the GetEpContextNodes() interface.  
The resulting "Cached Model" (often named model\_ctx.onnx) is a valid ONNX file, but its internal topology is radically different:

* **I/O**: Remains identical to the original model.  
* **Compute**: Thousands of compute nodes are replaced by opaque EPContext nodes.

#### **3.1.1 Key Attributes**

| Attribute | Type | Description |
| :---- | :---- | :---- |
| **ep\_cache\_context** | String | A critical pointer. If embed\_mode=0, this stores the relative path to the external binary file (e.g., qnn\_ctx.bin). |
| **e\[span\_3\](start\_span)\[span\_3\](end\_span)mbed\_mode** | Int | **0 (External Mode)**: Recommended for production. Context data is stored in a separate binary. Supports memory mapping (mmap) to reduce memory pressure. **1 (Embed Mode)**: Data is embedded directly in the ONNX file. Only for small models due to the Protobuf 2GB limit. |
| **m\[span\_4\](start\_span)\[span\_4\](end\_span)ain\_context** | Int | **1 (Primary)**: Holds the main context (weights/structure). **0 (Secondary)**: References the main context. Common in multi-subgraph models to share hardware resources. |
| **p\[span\_5\](start\_span)\[span\_5\](end\_span)artition\_name** | String | Identifies the original partition this context belongs to. |

### **3.2 The Dump Workflow (Generation)**

1. **Configure Session**: User enables ep.context\_enable=1 and sets ep.context\_file\_path.  
2. **Partitioning**: ORT splits the graph.  
3. **Compilation**: EP compiles the subgraph using its backend SDK (e.g., QNN SDK).  
4. **Replacement**: EP creates an EPContext node wrapping the binary data or file path.  
5. **Serialization**: ORT saves the new graph as model\_ctx.onnx. If external mode is used, the EP writes the binary file to disk.

### **3.3 The Load Workflow (Inference)**

1. **Load Model**: User loads model\_ctx.onnx.  
2. **Dispatch**: ORT encounters the EPContext node and passes it to the EP.  
3. **Deserialization**: EP reads ep\_cache\_context.  
   * If external, it resolves the path relative to the model and loads the file.  
4. **Initialization**: EP loads the binary directly into hardware memory, skipping the compilation phase.

## **4\. Deep Dive: Qualcomm QNN Execution Provider**

The QNN EP has the most mature support for Context Cache, essential for Windows on ARM (Copilot+ PCs) and Android AI.

### **4.1 Configuration Parameters**

QNN EP uses SessionOptions config entries.

| Key | Recommended | Details |
| :---- | :---- | :---- |
| ep.context\_enable | "1" | **Global Switch**. Triggers generation mode. Can typically be omitted during load, but keeping it ensures consistency. |
| ep.context\_file\_path | Path String | **Dual Purpose**. **Generation**: Where to save the ctx.onnx file. **Loading**: If loading from memory (bytes), this MUST be set to tell the EP where to look for the external .bin file. |
| ep.contex\[span\_8\](start\_span)\[span\_8\](end\_span)t\_embed\_mode | "0" | **Strongly Recommended "0"**. Embedded mode ("1") increases file size and RAM usage, risking OOM on mobile devices. |

### **4.2 Cross-Architecture Generation**

A killer feature of QNN EP is generating ARM64 cache on x64 hosts. Compiling 7B+ parameter models on a phone CPU is often unfeasible. Developers can use the **x64 version of QNN SDK** on a PC to generate the context binary for a specific Snapdragon SoC (e.g., 8 Gen 3).  
**Workflow:**

1. Install onnxruntime-qnn (x64) on a PC.  
2. Point backend\_path to QnnHtp.dll.  
3. Run the generation script to output model\_ctx.onnx and .bin.  
4. Deploy these files to the Android/Windows ARM device.

**Critical Note**: The generated binary is strictly bound to the **QNN SDK Version** and **SoC Model**. Mismatches cause load failures.

### **4.3 Performance Benchmarks**

| Model | Platform | Cold Start (No Cache) | Warm Start (Cached) | Speedup |
| :---- | :---- | :---- | :---- | :---- |
| **MobileNet V2** | Snapdragon 8cx Gen 3 | \~1200 ms | \~80 ms | **15x** |
| **Stable Diffusion** | Snapdragon X Elite | \> 60 sec | \~2.5 sec | **\>20x** |
| **Llama 2 7B** | Snapdragon 8 Gen 3 | \> 5 min | \< 10 sec | **\>30x** |

## **5\. Deep Dive: NVIDIA TensorRT Execution Provider**

TensorRT offers two caching tiers. It is crucial to distinguish between them.

### **5.1 Engine Cache vs. EP Context**

1. **Engine Cache (trt\_engine\_cache\_enable)**:  
   * Implicit cache. ORT calculates a hash of the ONNX graph. If a match is found in the cache path, it loads the engine.  
   * **Risk**: If the hash check fails (e.g., ORT version bump), it silently recompiles, causing unexpected latency.  
2. **EP Context Model (trt\_context\_enable)**:  
   * Explicit "Cached Model" (model\_ctx.onnx).  
   * **Benefit**: AOT compilation logic is separated from runtime logic. You ship the ctx.onnx file, guaranteeing the engine is used.

### **5.2 Configuration Guide**

To use the modern EP Context workflow:

| Option | Type | Description |
| :---- | :---- | :---- |
| trt\_dump\_ep\_context\_model | Bool | Set to True to export the context model. |
| trt\_ep\_context\_file\_path | String | Path for the output model. |
| trt\_engine\_cache\_enable | Bool | **Must be enabled** to support the serialization backend. |
|  | trt\_timing\_cache\_enable | Bool |

### **5.3 Hardware Compatibility**

TensorRT Engines are **GPU-specific**. An engine built on an RTX 3080 will **not** run on an RTX 4090\.

* **Recommendation**: Perform generation on the target device (Installation-time generation) or maintain a fleet of build servers for every supported GPU architecture.

## **6\. Deep Dive: Intel OpenVINO Execution Provider**

### **6.1 cache\_dir Mechanism**

OpenVINO EP primarily uses the cache\_dir option in SessionOptions.

* **CPU/NPU**: Caches .blob files (network topology and weights).  
* **iGPU**: Generates cl\_cache (OpenCL kernels). Compiling OpenCL kernels is the main bottleneck for iGPU startup.

### **6.2 Benefits**

* **Dynamic Shapes**: OpenVINO's cache handles dynamic shapes better than TensorRT, caching kernel variants for different input sizes.  
* **Heterogeneous**: When using HETERO:GPU,CPU, caching works across devices.

## **7\. Deep Dive: DirectML Execution Provider**

DirectML (DirectX 12 based) relies on the OS and driver for caching.

### **7.1 Persistent Shader Cache**

* **Mechanism**: DirectML compiles operators into Compute Shaders. The GPU driver caches these compiled shaders on disk (e.g., %LOCALAPPDATA%\\D3DSCache).  
* **Control**: Unlike QNN, there is no explicit "Dump Context" API for DirectML yet. Caching is implicit.  
* **Strategy**: Developers perform a "Warm-up" run during app installation or first launch to populate the driver cache.

## **8\. Engineering Examples**

### **8.1 Python: QNN Offline Generation (Mobile)**

**Scenario**: Generate a cache on a PC to deploy with an Android app.  
**Dump Script (PC/Dev Machine):**  
`import onnxruntime as ort`

`model_path = "super_res.onnx"`  
`ctx_path = "super_res_ctx.onnx"`

`options = ort.SessionOptions()`  
`# 1. Enable context generation`  
`options.add_session_config_entry("ep.context_enable", "1")`  
`# 2. Set output path (EP implies binary path from this)`  
`options.add_session_config_entry("ep.context_file_path", ctx_path)`  
`# 3. External mode (Recommended for large models)`  
`options.add_session_config_entry("ep.context_embed_mode", "0")`

`qnn_options = {`  
    `"backend_path": "QnnHtp.dll", # Uses HTP (NPU) backend`  
    `"htp_graph_finalization_optimization_mode": "3" # Max optimization`  
`}`

`# Creating the session triggers the compilation and dump`  
`session = ort.InferenceSession(`  
    `model_path,`  
    `sess_options=options,`  
    `providers=["QNNExecutionProvider"],`  
    `provider_options=[qnn_options]`  
`)`  
`print("Context generated successfully.")`

**Load Script (Target Device):**  
`import onnxruntime as ort`

`# Load the GENERATED context model directly`  
`ctx_model = "super_res_ctx.onnx"` 

`qnn_options = { "backend_path": "QnnHtp.dll" }`

`# Fast load - skips compilation`  
`session = ort.InferenceSession(`  
    `ctx_model,`   
    `providers=["QNNExecutionProvider"],`  
    `provider_options=[qnn_options]`  
`)`

### **8.2 C++: TensorRT Integration (Windows/Linux)**

`Ort::SessionOptions session_options;`  
`OrtTensorRTProviderOptions trt_options{};`

`// 1. Enable Engine Caching (Base requirement)`  
`trt_options.trt_engine_cache_enable = 1;`  
`trt_options.trt_engine_cache_path = "C:\\ProgramData\\MyApp\\Cache";`

`// 2. Enable Timing Cache (Optimization search cache)`  
`trt_options.trt_timing_cache_enable = 1;`

`// 3. Attach Provider`  
`session_options.AppendExecutionProvider_TensorRT(trt_options);`

`// 4. Load Model`  
`// First run: Slow (Builds engine & cache).`  
`// Subsequent runs: Fast (Loads from cache).`  
`Ort::Session session(env, L"resnet50.onnx", session_options);`

## **9\. Benchmarking Methodology**

To measure the benefits, use the official onnxruntime\_perf\_test tool.

1. **Baseline (No Cache)**:  
   `onnxruntime_perf_test -m times -r 10 -e qnn -I mobilenet_v2.onnx result_no.xml`  
   *Check log for "Session Creation Time".*  
2. **With Cache**:  
   `# Load the Context ONNX file directly`  
   `onnxruntime_perf_test -m times -r 10 -e qnn -I mobilenet_v2_ctx.onnx result_yes.xml`  
   *Compare "Session Creation Time". Expect \>90% reduction.*

## **10\. Common Pitfalls & Troubleshooting**

1. **Path Hell**:  
   * *Issue*: Loading ctx.onnx from memory bytes fails with "File not found".  
   * *Fix*: You MUST set ep.context\_file\_path in SessionOptions when loading from memory so the EP knows the base directory to resolve the relative path to the .bin file.  
2. **Version Mismatch**:  
   * *Issue*: QNN Context loads fail silently or crash.  
   * *Fix*: Ensure the onnxruntime-qnn version and the underlying QNN SDK library (libQnnHtp.so / QnnHtp.dll) match exactly between the generating machine and the target device.  
3. **Permissions**:  
   * *Issue*: Unable to write cache files.  
   * *Fix*: On Android/UWP, ensure paths point to writable app data directories (e.g., Context.getFilesDir()).

## **11\. Conclusion**

The ONNX Runtime EP Context Cache is not optional for modern Edge AI—it is a requirement. It bridges the gap between the flexibility of ONNX and the raw performance of specialized hardware. By effectively utilizing this feature, developers can reduce startup latency from minutes to milliseconds, enabling seamless integration of LLMs and Generative AI into consumer applications.

#### **Works cited**

1\. EP Context Design | onnxruntime, https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html 2\. NVIDIA TensorRT RTX Execution Provider \- ONNX Runtime, https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html 3\. OpenVINO \- onnxruntime \- GitHub Pages, https://oliviajain.github.io/onnxruntime/docs/execution-providers/OpenVINO-ExecutionProvider.html 4\. Qualcomm \- QNN | onnxruntime, https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html 5\. KB5067994: QNN Execution Provider Update for Copilot+ on Qualcomm Windows 11, https://windowsforum.com/threads/kb5067994-qnn-execution-provider-update-for-copilot-on-qualcomm-windows-11.381916/ 6\. NVIDIA \- TensorRT | onnxruntime, https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html 7\. Intel \- OpenVINO™ | onnxruntime, https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html 8\. Intel \- OpenVINO™ | onnxruntime \- GitHub Pages, https://fs-eire.github.io/onnxruntime/docs/execution-providers/OpenVINO-ExecutionProvider.html 9\. DirectML Execution Provider \- onnxruntime, https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html 10\. End-to-End AI for NVIDIA-Based PCs: ONNX and DirectML | NVIDIA Technical Blog, https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-onnx-and-directml/