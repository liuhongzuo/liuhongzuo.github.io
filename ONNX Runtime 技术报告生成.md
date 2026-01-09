# **The Architecture and Implementation of ONNX Runtime: A Technical Report**

## **1\. Fundamental Concepts and Architectural Overview**

The rapid evolution of deep learning has resulted in a Cambrian explosion of software frameworks and hardware accelerators. While frameworks like PyTorch, TensorFlow, and Keras provide the necessary flexibility for model research and training—prioritizing features like automatic differentiation and dynamic computational graphs—they are often suboptimal for production deployment. Production environments require distinct characteristics: minimal latency, maximized throughput, strict memory constraints, and the ability to leverage diverse hardware backends ranging from edge NPUs to data center GPUs.

ONNX Runtime (ORT) was engineered to bridge this dichotomy. It is not merely an interpreter for the Open Neural Network Exchange (ONNX) format but a high-performance, cross-platform inference engine designed to unify the execution of machine learning models across disparate hardware.1

### **1.1 The ONNX Standard and Intermediate Representation**

At the core of ONNX Runtime is the ONNX format, a standard for representing machine learning models. Unlike framework-specific formats that may rely on language-specific serialization (like Python’s pickle in PyTorch), ONNX uses Protocol Buffers (protobuf) to define a language-agnostic schema.

An ONNX model is fundamentally a computation graph. This graph is composed of:

* **Nodes:** The computational operators (e.g., Conv2D, Relu, MatMul). Each node refers to an operator versioned within a specific "Opset," ensuring backward compatibility as the standard evolves.2  
* **Edges:** These represent the data flow dependencies between nodes. They carry **Tensors**, which are typed multi-dimensional arrays (e.g., Float32, Int8).  
* **Initializers:** Constant tensors, typically representing the trained weights and biases of the model.  
* **ValueInfo:** Metadata describing the shape and type of inputs and outputs at graph boundaries.

ONNX Runtime parses this protobuf definition into an in-memory graph representation (IR). This internal IR is the data structure upon which all optimizations, partitioning, and execution logic operate. The decoupling of the serialized file format from the in-memory execution graph is a critical architectural decision, allowing ORT to perform aggressive graph mutations without altering the original model file unless explicitly requested.1

### **1.2 High-Level System Architecture**

The architecture of ONNX Runtime is constructed around the principle of **Heterogeneous Execution**. It operates on the assumption that no single hardware accelerator can execute every operator in a complex neural network. Therefore, the runtime must act as an orchestrator, intelligently routing different parts of the computation graph to the most efficient available hardware.1

The system is composed of several hierarchical layers:

1. **The API Layer:** Provides language bindings (C++, Python, C\#, Java, C) that expose a consistent surface for model loading and execution.3  
2. **The Graph Engine:** Responsible for parsing the model, validating schemas against the operator registry, and applying provider-independent optimizations (such as constant folding).  
3. **The Partitioning Engine:** A sophisticated subsystem that queries registered Execution Providers (EPs) to determine their capabilities and splits the graph into subgraphs accordingly.1  
4. **The Execution Engine:** Manages the orchestration of kernels. It handles memory allocation via the Arena allocator, schedules tasks across thread pools, and manages the data transfer between different execution providers (e.g., copying a tensor from CPU memory to GPU memory when transitioning between a CPU node and a CUDA node).

### **1.3 Key Design Decisions**

Several foundational design decisions define the performance characteristics of ORT:

* **Stateless Kernels:** The implementation of operators (Kernels) are strictly stateless. The Compute() method of a kernel receives a context object containing inputs and outputs but does not retain state between calls. This allows a single kernel instance to be shared across multiple simultaneous inference sessions or threads, significantly reducing memory overhead in high-concurrency server scenarios.1  
* **Pluggable Execution Providers:** Hardware support is not monolithic. Instead, hardware vendors implement the IExecutionProvider interface. This allows ORT to support new accelerators (like a new NPU or FPGA) without modifying the core runtime code. The runtime interacts with these providers through a defined contract of Capability Discovery and Compilation.4  
* **Tensor Abstraction:** ORT uses a standardized tensor representation for runtime values. While Execution Providers may use opaque, hardware-specific memory layouts internally (e.g., NC/32HW on a DSP), they must convert data to the standard ORT representation at the boundaries of the subgraph to ensure interoperability.1

## **2\. Detailed Analysis of Features**

ONNX Runtime encompasses a vast array of features designed to optimize performance (latency and throughput), memory usage, and deployment flexibility. These features can be broadly categorized into Graph Optimizations, Quantization, Memory Management, and Execution Control.

### **2.1 Graph Optimizations**

Graph optimizations are transformations applied to the computation graph to reduce computational complexity or improve memory locality. ORT categorizes these into three levels, controlled via the SessionOptions.

#### **2.1.1 Level 1: Basic Graph Optimizations**

These are semantic-preserving transformations that are universally applicable, regardless of the hardware backend. They are enabled by default and include:

* **Constant Folding:** The runtime identifies subgraphs that rely entirely on constant initializers. These subgraphs are evaluated once during the session initialization phase, and the subgraph is replaced by the resulting constant tensor. This eliminates redundant computations at inference time.5  
* **Redundant Node Elimination:** Operators that do not change data (such as Identity, Dropout in inference mode, or Slice operations that cover the full tensor) are removed. This reduces the overhead of kernel launching and memory copying.  
* **Common Subexpression Elimination (CSE):** If the graph contains multiple identical nodes (same operator, same inputs) computing the same result, ORT merges them into a single node, fan-out the output to all downstream consumers.

#### **2.1.2 Level 2: Extended Graph Optimizations**

These optimizations involve more complex node fusions that are often specific to the CPU or CUDA execution providers. While they generally improve performance, they might theoretically alter precision slightly due to changes in floating-point accumulation order.

* **Operator Fusion:** The most significant optimization in this category. Examples include:  
  * **Conv-Add-Activation Fusion:** A sequence of Convolution, Bias Add, and ReLU is fused into a single kernel. This reduces memory bandwidth pressure because intermediate results (e.g., the output of the convolution) are kept in the register file or L1 cache rather than being written out to global memory before the Activation is read back in.5  
  * **GEMM Activation Fusion:** Similar to Convolution fusion, applied to General Matrix Multiplication (Dense layers).  
  * **LayerNormalization Fusion:** Fuses the multiple primitive arithmetic operations (Mean, Variance, Sub, Div, Scale, Bias) constituting Layer Norm into a single optimized kernel.

#### **2.1.3 Level 3: Layout Optimizations**

This level transforms the data layout of tensors to match the vectorization characteristics of the hardware.

* **NCHWc Transformation:** Standard ONNX models typically use NCHW (Batch, Channels, Height, Width) layout. However, modern CPUs with AVX-512 instructions often perform better with blocked layouts like NCHWc (where c might be 8 or 16). This optimization inserts Reorder nodes to transform the graph into this hardware-friendly layout, enabling vector intrinsics to process multiple channels in a single clock cycle.5

#### **2.1.4 Online vs. Offline Optimization**

Optimizations can be performed **Online** (every time the session initializes) or **Offline**.

* **Online:** Convenient but increases model startup time.  
* **Offline:** The user performs optimizations once and saves the optimized model to disk. This is critical for mobile deployment where startup latency and battery usage are paramount. However, offline models become tied to the specific hardware and ORT version used during optimization.5

### **2.2 Quantization**

Quantization is the process of mapping continuous infinite precision values (Floating Point) to a smaller, finite set of discrete values (typically Int8 or UInt8). ORT supports a comprehensive quantization suite.6

#### **2.2.1 Quantization Modes**

* **Dynamic Quantization:** The scaling factors (scale and zero-point) are calculated *during inference* based on the actual range of the activation tensors. This is computationally more expensive per inference but requires no calibration data. It is highly effective for Transformer-based models (like BERT) where activation ranges vary drastically between inputs.  
* **Static Quantization:** The scaling factors are pre-computed using a calibration dataset. The model weights and activations are quantized offline. During inference, no range calculation is needed. This offers the best performance benefit but requires a representative dataset to maintain accuracy.  
* **Quantization-Aware Training (QAT):** ORT supports running models that were trained with simulated quantization (fake quantization nodes). This typically yields the highest accuracy for low-precision inference as the network learns to adapt to the rounding errors.6

#### **2.2.2 Hardware Acceleration for Quantization**

ORT leverages hardware-specific instructions for quantized math:

* **x64:** Uses VNNI (Vector Neural Network Instructions) on newer Intel CPUs.  
* **ARM:** Uses NEON instructions or dot-product extensions.  
* **NVIDIA:** Uses Tensor Cores for Int8 arithmetic.

### **2.3 Memory Management: The Arena Allocator**

Efficient memory management is a prerequisite for high-performance inference. Repeatedly calling system allocators (malloc/free or cudaMalloc) introduces significant overhead and fragmentation.

* **The Arena Strategy:** ORT allocates a large contiguous block of memory (the Arena) upfront. During execution, it sub-allocates chunks from this block to tensor kernels.  
* **BFCA (Best Fit with Coalescing):** ORT employs a specialized allocator strategy called Best Fit with Coalescing. When a tensor is freed, its chunk is coalesced with adjacent free chunks to form larger blocks, reducing fragmentation.7  
* **Memory Pattern Planning:** For static models (where input shapes are fixed), ORT analyzes the entire execution plan during initialization. It determines the lifespan of every intermediate tensor. If Tensor A is needed only for Node 1 and Tensor B is needed only for Node 2, they can share the same physical memory address (offset) within the Arena. This dramatically reduces the **Peak Memory Footprint** of the model.7  
* **Arena Shrinkage:** In dynamic scenarios (e.g., a server handling variable-length text), the Arena might grow to accommodate a uniquely large request. To prevent this "high water mark" from persisting indefinitely (wasting RAM), ORT supports an Arena Shrinkage mechanism that periodically trims unused memory pages back to the OS.7

### **2.4 Threading and Parallelism**

ORT provides a sophisticated threading model to maximize CPU utilization.8

| Threading Mode | Description | Use Case |
| :---- | :---- | :---- |
| **Intra-Op Parallelism** | Splits a single operator's workload across multiple threads. For example, a large Matrix Multiplication is tiled, and different tiles are computed by different threads. | Low latency scenarios (Batch Size \= 1). Reducing the time for a single request. |
| **Inter-Op Parallelism** | Executes independent graph branches concurrently. If a model has two parallel towers (e.g., Inception blocks), they can be run simultaneously on different threads. | High throughput scenarios. Utilizing many cores when the graph structure allows it. |

ORT allows users to configure the size of these thread pools and set thread affinity (pinning threads to specific cores) to avoid cache thrashing.

### **2.5 I/O Binding**

A common bottleneck in high-performance pipelines is the data copy between CPU and GPU. By default, session.Run() accepts CPU arrays, copies them to the GPU, executes, and copies results back.

**I/O Binding** allows the user to pre-allocate memory on the device (e.g., GPU memory) and pass pointers to this device memory directly to ORT.

* **Scenario:** A video decoding pipeline runs on the GPU (NVDEC). The decoded frames are already in VRAM. Using I/O Binding, ORT can read these frames directly without a round-trip to the CPU, significantly reducing latency and PCIe bandwidth usage.3

### **2.6 Custom Operators**

While ONNX has a rich operator set, research often requires novel operations not yet standardized. ORT supports **Custom Operators**, allowing users to write C++ kernels for new ops and register them with the runtime. These custom ops can be compiled into a shared library and loaded dynamically, ensuring that the main ORT binary does not need recompilation.9

## **3\. Inference Implementation Example**

This section details the implementation of an inference pipeline using the ONNX Runtime C++ API. C++ is chosen for this example as it represents the standard for production deployments where performance is critical.

### **3.1 Prerequisites**

To execute this code, the following components are required:

* **Headers:** onnxruntime\_cxx\_api.h (The primary C++ wrapper).  
* **Binaries:** onnxruntime.dll (Windows) or libonnxruntime.so (Linux).  
* **Model:** A serialized ONNX file (e.g., squeezenet.onnx).

### **3.2 Detailed C++ Code Example**

C++

\#**include** \<onnxruntime\_cxx\_api.h\>  
\#**include** \<iostream\>  
\#**include** \<vector\>  
\#**include** \<numeric\>  
\#**include** \<algorithm\>

// Function to calculate the total number of elements from a shape vector  
int64\_t CalculateElementCount(const std::vector\<int64\_t\>& shape) {  
    if (shape.empty()) return 0;  
    int64\_t size \= 1;  
    for (auto dim : shape) {  
        // Handling dynamic dimensions (often \-1) by assuming a batch size of 1 for demo  
        if (dim \< 0) return 1 \* size;   
        size \*= dim;  
    }  
    return size;  
}

int main() {  
    // 1\. Initialize the ORT Environment  
    // The environment retains global state such as thread pools and logging configuration.  
    // It is thread-safe and usually created once per process.  
    Ort::Env env(ORT\_LOGGING\_LEVEL\_WARNING, "ConsoleInferenceApp");

    // 2\. Configure Session Options  
    Ort::SessionOptions session\_options;  
      
    // Set Graph Optimization Level to MAX (Level 3 \- Layout Optimizations)  
    session\_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT\_ENABLE\_ALL);  
      
    // Configure Threading: Use 4 threads for parallelizing within operators  
    session\_options.SetIntraOpNumThreads(4);

    // Append Execution Provider (Optional)  
    // This line would enable CUDA if the GPU package is used.  
    // OrtSessionOptionsAppendExecutionProvider\_CUDA(session\_options, 0);

    // 3\. Load the Model and Create Session  
    // The construction of the session triggers model loading, validation, and optimization.  
\#**ifdef** \_WIN32  
    const wchar\_t\* model\_path \= L"squeezenet.onnx";  
\#**else**  
    const char\* model\_path \= "squeezenet.onnx";  
\#**endif**

    try {  
        Ort::Session session(env, model\_path, session\_options);

        // 4\. Inspect Input/Output Metadata  
        // An allocator is required to retrieve the names of the inputs/outputs  
        Ort::AllocatorWithDefaultOptions allocator;

        // Get Input Info  
        size\_t num\_input\_nodes \= session.GetInputCount();  
          
        // ORT requires input names to be passed into Run(). We must keep the string pointers valid.  
        // GetInputNameAllocated returns a smart pointer ensuring memory is freed.  
        auto input\_name \= session.GetInputNameAllocated(0, allocator);  
        const char\* input\_name\_ptr \= input\_name.get(); // Raw pointer for the Run API

        // Get Input Shape  
        Ort::TypeInfo input\_type\_info \= session.GetInputTypeInfo(0);  
        auto input\_tensor\_info \= input\_type\_info.GetTensorTypeAndShapeInfo();  
        std::vector\<int64\_t\> input\_shape \= input\_tensor\_info.GetShape();

        // Fix dynamic batch size: If the model expects \[-1, 3, 224, 224\], we set it to   
        if (input\_shape.size() \> 0 && input\_shape \== \-1) {  
            input\_shape \= 1;  
        }

        size\_t input\_tensor\_size \= CalculateElementCount(input\_shape);  
          
        // 5\. Prepare Input Data  
        // In a real application, this vector would be filled with image pixel data.  
        std::vector\<float\> input\_tensor\_values(input\_tensor\_size);  
        // Initialize with dummy data (0.0, 1.0, 2.0...)  
        std::iota(input\_tensor\_values.begin(), input\_tensor\_values.end(), 0.0f);

        // 6\. Create Input Tensor  
        // We create a tensor that wraps the existing data buffer. This is a Zero-Copy operation.  
        auto memory\_info \= Ort::MemoryInfo::CreateCpu(  
            OrtAllocatorType::OrtArenaAllocator,   
            OrtMemType::OrtMemTypeDefault  
        );

        Ort::Value input\_tensor \= Ort::Value::CreateTensor\<float\>(  
            memory\_info,   
            input\_tensor\_values.data(),   
            input\_tensor\_size,   
            input\_shape.data(),   
            input\_shape.size()  
        );

        // 7\. Prepare Output Tracking  
        // Get Output Name  
        size\_t num\_output\_nodes \= session.GetOutputCount();  
        auto output\_name \= session.GetOutputNameAllocated(0, allocator);  
        const char\* output\_name\_ptr \= output\_name.get();

        const char\* input\_names \= { input\_name\_ptr };  
        const char\* output\_names \= { output\_name\_ptr };

        // 8\. Run Inference  
        // The Run() call is the synchronization point. It executes the graph.  
        // We pass '1' for number of inputs and '1' for number of outputs.  
        auto output\_tensors \= session.Run(  
            Ort::RunOptions{nullptr},   
            input\_names,   
            \&input\_tensor,   
            1,   
            output\_names,   
            1  
        );

        // 9\. Process Results  
        // Get pointer to the raw output data (usually logits or probabilities)  
        float\* floatarr \= output\_tensors.GetTensorMutableData\<float\>();  
          
        // Print the first 5 output values  
        std::cout \<\< "Inference Result (first 5 elements): ";  
        for (int i \= 0; i \< 5; i++) {  
            std::cout \<\< floatarr\[i\] \<\< " ";  
        }  
        std::cout \<\< std::endl;

    } catch (const Ort::Exception& e) {  
        std::cerr \<\< "ONNX Runtime Exception: " \<\< e.what() \<\< std::endl;  
        return \-1;  
    }

    return 0;  
}

### **3.3 Analysis of the Example**

* **Environment (Ort::Env):** Represents the "process" level of the runtime. Creating and destroying this object frequently is an anti-pattern. It should be created once.  
* **Session Options:** This is where performance tuning happens. The SetIntraOpNumThreads call determines the size of the thread pool used for compute-heavy ops like MatMul. Setting this too high can cause context switching overhead; setting it too low underutilizes the CPU.  
* **Memory Info:** The Ort::MemoryInfo::CreateCpu call is critical. It tells ORT that the data we are providing (input\_tensor\_values) resides in standard CPU RAM (OrtMemTypeDefault). If we were using CUDA, we could define memory info representing GPU RAM to facilitate zero-copy from device pointers.3  
* **Tensor Creation:** Ort::Value::CreateTensor does *not* allocate new memory for the data; it creates a view over the std::vector. This implies that the std::vector must outlive the Ort::Value object, or undefined behavior will occur (dangling pointer).9

## **4\. Execution Provider (EP): Detailed Analysis and Implementation**

The **Execution Provider** abstraction is the architectural cornerstone that enables ONNX Runtime to support heterogeneous hardware. It allows the runtime to offload specific nodes or entire subgraphs to specialized hardware accelerators (e.g., NVIDIA GPUs via TensorRT, Intel CPUs via OpenVINO, Qualcomm NPUs via QNN).1

### **4.1 The Role of an Execution Provider**

An EP is a bridge between the ONNX graph representation and the hardware-specific execution primitives. It has three primary responsibilities:

1. **Capability Discovery:** Informing ORT which ONNX nodes it can execute.  
2. **Compilation (Optional):** compiling supported subgraphs into hardware-specific binary blobs (e.g., a TensorRT engine).  
3. **Execution:** Marshalling inputs, invoking the hardware kernel/blob, and marshalling outputs back to ORT.

### **4.2 Key Interfaces and Data Structures**

To implement an EP, one must inherit from the IExecutionProvider class defined in onnxruntime/core/framework/execution\_provider.h. Below is a detailed analysis of the critical virtual methods and associated data structures required for implementation.

#### **4.2.1 GetCapability: The Partitioning Handshake**

This is the first method invoked by ORT during the session initialization.

**Signature:**

C++

virtual std::vector\<std::unique\_ptr\<ComputeCapability\>\> GetCapability(  
    const onnxruntime::GraphViewer& graph\_viewer,  
    const IKernelLookup& kernel\_lookup,  
    const GraphOptimizerRegistry& graph\_optimizer\_registry,  
    IResourceAccountant\* resource\_accountant \= nullptr  
) const;

11

**Detailed Analysis:**

* **GraphViewer:** This object provides a read-only traversal interface for the ONNX graph. The EP iterates over the nodes in topological order.  
* **The Logic:** The EP developer writes logic to inspect each node. For example, a hypothetical "NPU\_EP" might inspect a Conv2D node. It checks:  
  * Is the kernel size supported (e.g., 3x3 or 5x5)?  
  * Are the strides supported?  
  * Is the input data type supported (e.g., only Float16)?  
* **Subgraph Fusion:** The EP is not limited to selecting individual nodes. It can identify patterns (e.g., Conv \+ Add \+ Relu) and claim the entire sequence.  
* **Return Value (ComputeCapability):** The method returns a vector of capabilities. Each capability wraps an IndexedSubGraph.  
  * **IndexedSubGraph:** A struct containing a list of NodeIndex integers. These indices tell ORT: "I claim these nodes."  
  * **Functionality:** If an EP returns a capability containing nodes {1, 2, 3}, ORT's partitioner will physically remove nodes 1, 2, and 3 from the main graph and replace them with a single "Fused Node" that targets this EP.

#### **4.2.2 Compile: Just-In-Time Compilation**

If the EP supports graph compilation (like TensorRT or OpenVINO), it implements the Compile method. This method is called *after* GetCapability and *after* the graph has been partitioned.

**Signature:**

C++

virtual common::Status Compile(  
    const std::vector\<FusedNodeAndGraph\>& fused\_nodes\_and\_graphs,  
    std::vector\<NodeComputeInfo\>& node\_compute\_funcs  
);

12

**Detailed Analysis:**

* **FusedNodeAndGraph:** This input structure pairs the specific fused node created by ORT with the subgraph (the "island" of nodes) that was assigned to it.  
* **Compilation Workflow:**  
  1. The EP iterates through the fused\_nodes\_and\_graphs.  
  2. For each subgraph, it translates the ONNX nodes into its own proprietary Intermediate Representation (e.g., building an nvinfer1::INetworkDefinition for TensorRT).  
  3. It invokes its backend compiler to generate an optimized binary (e.g., a TRT Engine or OpenVINO Blob).  
  4. It serializes this binary or stores it in memory, mapped to the fused node's name.  
* **NodeComputeInfo:** The EP returns this struct to ORT. It contains function pointers that ORT will call at runtime:  
  * create\_state\_func: Called when the session state is created. Used to allocate execution contexts (e.g., GPU memory for the engine).  
  * compute\_func: The actual execution entry point called during session.Run().  
  * release\_state\_func: Called upon session destruction to free resources.

#### **4.2.3 Compute: Execution Mechanism**

For EPs that do not use the Compile interface (i.e., they provide a library of pre-written kernels like the CUDA EP), the execution logic is defined in the Kernel Registry.

**The Kernel Contract:**

C++

class MyCustomKernel : public OpKernel {  
public:  
    Status Compute(OpKernelContext\* context) const override {  
        // 1\. Retrieve Inputs via context-\>Input\<Tensor\>(index)  
        // 2\. Compute (stateless)  
        // 3\. Write Output via context-\>Output(index, shape)  
        return Status::OK();  
    }  
};

**Analysis:**

* **Statelessness:** The Compute method is const. A kernel instance cannot mutate its own member variables during execution. This enforces thread safety, allowing the same kernel object to serve multiple concurrent inference requests. Any state required for a specific inference (like temporary buffers) must be allocated via the OpKernelContext.  
* **Memory Access:** The context abstracts the memory location. Whether the tensor is on CPU or GPU, the context provides a pointer. The EP logic must ensure it launches the correct CUDA/Assembly instructions for that memory type.1

### **4.3 Implementing a Custom EP: Step-by-Step**

To implement a new EP (e.g., "MyAcceleratorEP"), one follows this roadmap:

1. **Directory Structure:** Create onnxruntime/core/providers/my\_accelerator.  
2. **Header Definition:** Define class MyAcceleratorExecutionProvider : public IExecutionProvider.  
3. **Factory Implementation:** Implement a Provider Factory (C interface) that allows the API to instantiate the provider.4  
4. **Register Kernel/Capability:**  
   * If library-based: Implement GetKernelRegistry() to return a list of supported operators and their C++ kernel implementations.  
   * If compiler-based: Implement GetCapability() and Compile().  
5. **Build System Integration:** Add the new provider to onnxruntime\_providers.cmake so it is compiled into the ORT binary.  
6. **Allocator Integration:** If the hardware has its own memory (e.g., an FPGA with on-board DRAM), implement GetAllocator() to expose this memory to ORT. This allows ORT to allocate input/output tensors directly on the device.1

### **4.4 The Partitioning Algorithm**

ORT uses a "Greedy" partitioning strategy based on provider priority.

1. **Priority List:** The user provides a list of EPs, e.g., \`\`.  
2. **Pass 1 (TensorRT):** ORT calls GetCapability on TensorRT. TRT claims the nodes it supports (e.g., the heavy Conv2D layers). These are marked.  
3. **Pass 2 (CUDA):** ORT traverses the *remaining* unclaimed nodes and calls GetCapability on the CUDA EP. CUDA claims what it can (e.g., nodes TRT didn't support).  
4. **Fallback (CPU):** Any nodes remaining after all EPs have been queried are assigned to the default CPU EP.  
5. **Memcpy Insertion:** ORT analyzes the data flow between partitions. If a tensor flows from a CPU node to a TensorRT node, ORT automatically injects a MemcpyHostToDevice operator. This ensures seamless execution without the user manually managing memory transfers.1

## **5\. Summary**

ONNX Runtime architecture effectively decouples the model definition (ONNX) from the execution mechanics (Execution Providers). By rigorously defining the interfaces for GetCapability and Compile, ORT allows hardware vendors to plug in sophisticated compilers while maintaining a unified user API. The layered approach—spanning basic graph optimizations to complex partitioning and JIT compilation—ensures that models are executed with maximum efficiency on whatever hardware is available, fulfilling the promise of "Train Once, Run Anywhere."

#### **引用的著作**

1. ONNX Runtime Architecture, 访问时间为 一月 9, 2026， [https://onnxruntime.ai/docs/reference/high-level-design.html](https://onnxruntime.ai/docs/reference/high-level-design.html)  
2. ONNX Concepts \- ONNX 1.21.0 documentation, 访问时间为 一月 9, 2026， [https://onnx.ai/onnx/intro/concepts.html](https://onnx.ai/onnx/intro/concepts.html)  
3. Python API documentation \- ONNX Runtime, 访问时间为 一月 9, 2026， [https://onnxruntime.ai/docs/api/python/api\_summary.html](https://onnxruntime.ai/docs/api/python/api_summary.html)  
4. Add a new Execution Provider to ONNX Runtime, 访问时间为 一月 9, 2026， [https://onnxruntime.ai/docs/execution-providers/add-execution-provider.html](https://onnxruntime.ai/docs/execution-providers/add-execution-provider.html)  
5. Graph Optimizations in ONNX Runtime, 访问时间为 一月 9, 2026， [https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)  
6. Quantize ONNX models | onnxruntime, 访问时间为 一月 9, 2026， [https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)  
7. C | onnxruntime, 访问时间为 一月 9, 2026， [https://onnxruntime.ai/docs/get-started/with-c.html](https://onnxruntime.ai/docs/get-started/with-c.html)  
8. Class SessionOptions \- ONNX Runtime, 访问时间为 一月 9, 2026， [https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.SessionOptions.html](https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.SessionOptions.html)  
9. Custom operators | onnxruntime, 访问时间为 一月 9, 2026， [https://onnxruntime.ai/docs/reference/operators/add-custom-op.html](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)  
10. QNN Execution Provider \- Qualcomm \- ONNX Runtime, 访问时间为 一月 9, 2026， [https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)  
11. onnxruntime/include/onnxruntime/core/framework/execution\_provider.h at main · microsoft/onnxruntime \- GitHub, 访问时间为 一月 9, 2026， [https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/framework/execution\_provider.h](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/framework/execution_provider.h)  
12. onnxruntime/onnxruntime/core/providers/dnnl/dnnl\_execution\_provider.h at main \- GitHub, 访问时间为 一月 9, 2026， [https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/dnnl/dnnl\_execution\_provider.h](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/dnnl/dnnl_execution_provider.h)