import onnx
from onnx import helper
import os

def wrap_nnc_to_epcontext(
    original_onnx_path: str,
    compiled_nnc_path: str,
    output_onnx_path: str,
    ep_domain: str = "com.microsoft"
):
    """
    Parses inputs and outputs from an original ONNX model and wraps a pre-compiled 
    .nnc binary file into a new ONNX model containing only an EPContext node.
    """
    
    # 1. Load the original ONNX model to extract input/output metadata
    if not os.path.exists(original_onnx_path):
        raise FileNotFoundError(f"Original ONNX model not found: {original_onnx_path}")
    
    print(f"[*] Loading original model: {original_onnx_path}")
    # load_external_data=False speeds up loading since we only need the graph structure, not the weights
    original_model = onnx.load(original_onnx_path, load_external_data=False)
    
    # Deep copy the input and output ValueInfoProtos to ensure they are cleanly 
    # detached from the original model's graph
    inputs = [onnx.ValueInfoProto.FromString(i.SerializeToString()) for i in original_model.graph.input]
    outputs = [onnx.ValueInfoProto.FromString(o.SerializeToString()) for o in original_model.graph.output]
    
    # Extract just the names for the node connections
    input_names = [i.name for i in inputs]
    output_names = [o.name for o in outputs]

    # 2. Read the pre-compiled .nnc binary file
    if not os.path.exists(compiled_nnc_path):
        raise FileNotFoundError(f"Compiled binary not found: {compiled_nnc_path}")
        
    print(f"[*] Reading compiled binary: {compiled_nnc_path}")
    with open(compiled_nnc_path, "rb") as f:
        binary_data = f.read()

    # 3. Create the EPContext node
    # Map the inputs and outputs exactly as they appeared in the original model
    ep_node = helper.make_node(
        op_type='EPContext',
        inputs=input_names,
        outputs=output_names,
        name='EPContext_Node',
        domain=ep_domain,
        # Attributes required by the Execution Provider
        partition_name='ep_partition_0',
        # Inject the binary data as a string attribute
        # Note: Change 'ep_cache_context' if your specific EP requires a different attribute name
        ep_cache_context=binary_data 
    )

    # 4. Create a new graph containing ONLY the EPContext node
    new_graph = helper.make_graph(
        nodes=[ep_node],
        name='Wrapped_EPContext_Graph',
        inputs=inputs,
        outputs=outputs
    )

    # 5. Define Opset Imports
    # We must declare both the standard ONNX opset and the custom domain opset
    onnx_opset = onnx.OperatorSetIdProto()
    onnx_opset.domain = ""
    onnx_opset.version = original_model.opset_import[0].version if original_model.opset_import else 14

    ms_opset = onnx.OperatorSetIdProto()
    ms_opset.domain = ep_domain
    ms_opset.version = 1

    # 6. Create the final ONNX model
    new_model = helper.make_model(
        new_graph,
        producer_name='NNC_EPContext_Wrapper',
        opset_imports=[onnx_opset, ms_opset]
    )
    new_model.ir_version = original_model.ir_version

    # 7. Save the new model, saving the large binary as external data to avoid the 2GB protobuf limit
    print(f"[*] Saving new model to: {output_onnx_path}")
    external_data_file = os.path.basename(output_onnx_path) + ".data"
    
    onnx.save_model(
        new_model,
        output_onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_file,
        size_threshold=1024 # Automatically push data larger than 1KB to the external file
    )
    
    print(f"[+] Done! Successfully created {output_onnx_path} and {external_data_file}")

if __name__ == "__main__":
    # Example execution
    # Replace these paths with your actual files when running
    original_model_path = "path/to/your_original_model.onnx"
    nnc_binary_path = "path/to/your_compiled_model.nnc"
    output_path = "path/to/final_epcontext_model.onnx"
    
    # Uncomment the line below to run the function
    # wrap_nnc_to_epcontext(original_model_path, nnc_binary_path, output_path)
