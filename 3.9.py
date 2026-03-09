import onnx
from onnx import helper, TensorProto
import json
import os

# Mapping string types from JSON to ONNX TensorProto enum types
TYPE_MAP = {
    "float32": TensorProto.FLOAT,
    "float16": TensorProto.FLOAT16,
    "float64": TensorProto.DOUBLE,
    "int64": TensorProto.INT64,
    "int32": TensorProto.INT32,
    "int8": TensorProto.INT8,
    "uint8": TensorProto.UINT8,
    "bool": TensorProto.BOOL
}

def create_value_info_from_dict(tensor_dict: dict) -> onnx.ValueInfoProto:
    """
    Helper function to create a ValueInfoProto from a dictionary parsed from JSON.
    """
    name = tensor_dict["name"]
    type_str = tensor_dict["type"].lower()
    shape = tensor_dict["shape"]
    
    if type_str not in TYPE_MAP:
        raise ValueError(f"Unsupported data type '{type_str}' for tensor '{name}'")
    
    onnx_type = TYPE_MAP[type_str]
    
    return helper.make_tensor_value_info(name, onnx_type, shape)

def wrap_nnc_with_json_config(
    json_config_path: str,
    compiled_nnc_path: str,
    output_onnx_path: str
):
    """
    Reads I/O configurations from a JSON file and wraps a pre-compiled 
    .nnc binary into an ONNX model with an EPContext node.
    """
    
    # =====================================================================
    # 1. Load and Parse JSON Configuration
    # =====================================================================
    print(f"[*] Loading configuration from: {json_config_path}")
    with open(json_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    inputs_config = config.get("inputs", [])
    outputs_config = config.get("outputs", [])
    ep_settings = config.get("ep_settings", {})
    
    ep_domain = ep_settings.get("domain", "com.microsoft")
    partition_name = ep_settings.get("partition_name", "ep_partition_0")
    payload_attr_name = ep_settings.get("payload_attribute_name", "ep_cache_context")

    # =====================================================================
    # 2. Dynamically Construct Inputs and Outputs
    # =====================================================================
    inputs = [create_value_info_from_dict(inp) for inp in inputs_config]
    outputs = [create_value_info_from_dict(out) for out in outputs_config]
    
    # Extract names for the EPContext node's I/O connections
    input_names = [i.name for i in inputs]
    output_names = [o.name for o in outputs]
    
    print(f"[*] Configured {len(inputs)} input(s) and {len(outputs)} output(s).")

    # =====================================================================
    # 3. Read the Pre-compiled Binary
    # =====================================================================
    if not os.path.exists(compiled_nnc_path):
        raise FileNotFoundError(f"Compiled binary not found: {compiled_nnc_path}")
        
    print(f"[*] Reading compiled binary: {compiled_nnc_path}")
    with open(compiled_nnc_path, "rb") as f:
        binary_data = f.read()

    # =====================================================================
    # 4. Create the EPContext Node dynamically
    # =====================================================================
    # We use dictionary unpacking (**node_attributes) to dynamically set attributes
    # based on the JSON configuration.
    node_attributes = {
        "partition_name": partition_name,
        payload_attr_name: binary_data 
    }
    
    ep_node = helper.make_node(
        op_type='EPContext',
        inputs=input_names,
        outputs=output_names,
        name='EPContext_Node',
        domain=ep_domain,
        **node_attributes  # Unpack attributes here
    )

    # =====================================================================
    # 5. Build Graph and Model
    # =====================================================================
    new_graph = helper.make_graph(
        nodes=[ep_node],
        name='Wrapped_EPContext_Graph',
        inputs=inputs,
        outputs=outputs
    )

    # Set up opsets
    onnx_opset = onnx.OperatorSetIdProto()
    onnx_opset.domain = ""
    onnx_opset.version = 14

    ms_opset = onnx.OperatorSetIdProto()
    ms_opset.domain = ep_domain
    ms_opset.version = 1

    new_model = helper.make_model(
        new_graph,
        producer_name='JSON_Driven_EPContext_Wrapper',
        opset_imports=[onnx_opset, ms_opset]
    )
    new_model.ir_version = 8

    # =====================================================================
    # 6. Save Model with External Data
    # =====================================================================
    print(f"[*] Saving new model to: {output_onnx_path}")
    external_data_file = os.path.basename(output_onnx_path) + ".data"
    
    onnx.save_model(
        new_model,
        output_onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_file,
        size_threshold=1024
    )
    
    print(f"[+] Done! Successfully created {output_onnx_path} and {external_data_file}")


if __name__ == "__main__":
    # Example execution
    json_path = "model_config.json"
    nnc_path = "test_model.nnc"
    out_onnx_path = "final_json_epcontext.onnx"
    
    # For testing: Generate a dummy .nnc file if it doesn't exist
    if not os.path.exists(nnc_path):
        with open(nnc_path, "wb") as f:
            f.write(b"mock_hardware_binary_data")
            
    # Make sure you have the model_config.json saved in the same directory before running
    if os.path.exists(json_path):
        wrap_nnc_with_json_config(json_path, nnc_path, out_onnx_path)
    else:
        print(f"[-] Please create {json_path} first to test the script.")
