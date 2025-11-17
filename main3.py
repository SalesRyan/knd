import flatbuffers
import importlib
import tflite
import sys

def get_mod(name):
    return importlib.import_module(name)

def vec_int(builder, values):
    if not values:
        return 0
    builder.StartVector(4, len(values), 4)
    for v in reversed(values):
        builder.PrependInt32(v)
    return builder.EndVector()

def create_buffer_offset(builder, data_bytes):
    # Use CreateByteVector (which is safe to call anytime when not inside a table)
    if not data_bytes:
        # create empty buffer (no data)
        tflite.Buffer.BufferStart(builder)
        buf_off = tflite.Buffer.BufferEnd(builder)
        return buf_off
    data_vec = builder.CreateByteVector(data_bytes)
    tflite.Buffer.BufferStart(builder)
    tflite.Buffer.BufferAddData(builder, data_vec)
    buf_off = tflite.Buffer.BufferEnd(builder)
    return buf_off

def create_tensor_offset(builder, name_str, shape_list, ttype, buffer_idx):
    # create name and shape vector BEFORE starting the Tensor table
    name_off = builder.CreateString(name_str) if name_str else 0
    shape_vec = vec_int(builder, shape_list) if shape_list else 0

    tflite.Tensor.TensorStart(builder)
    if shape_vec:
        tflite.Tensor.TensorAddShape(builder, shape_vec)
    tflite.Tensor.TensorAddType(builder, ttype)
    tflite.Tensor.TensorAddBuffer(builder, buffer_idx)
    if name_off:
        tflite.Tensor.TensorAddName(builder, name_off)
    return tflite.Tensor.TensorEnd(builder)

def inject(input_path, output_path):
    data = open(input_path, "rb").read()

    # import Model class safely
    try:
        model_mod = importlib.import_module("tflite.Model")
        ModelGetRoot = getattr(model_mod, "Model").GetRootAsModel
    except Exception:
        # fallback
        ModelGetRoot = tflite.Model.Model.GetRootAsModel

    model = ModelGetRoot(data, 0)

    # prepare builder with enough size
    builder = flatbuffers.Builder(max(1024, len(data) * 2))

    # --- BUFFERS: collect raw bytes from original model and create buffer offsets ---
    orig_buffers = []
    for i in range(model.BuffersLength()):
        b = model.Buffers(i)
        # get raw bytes (DataAsNumpy may exist)
        raw = None
        if hasattr(b, "DataAsNumpy"):
            raw = b.DataAsNumpy()
        else:
            try:
                raw = b.Data()
            except Exception:
                raw = None
        if raw is None:
            orig_buffers.append(None)
        else:
            try:
                orig_buffers.append(bytes(raw))
            except Exception:
                orig_buffers.append(raw.tobytes())

    buffer_offsets = []
    for rb in orig_buffers:
        buffer_offsets.append(create_buffer_offset(builder, rb))

    # --- OPERATOR CODES ---
    opcode_offsets = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        # prepare custom_code string BEFORE starting OperatorCode table
        custom_off = builder.CreateString(oc.CustomCode().decode()) if oc.CustomCode() else 0
        tflite.OperatorCode.OperatorCodeStart(builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(builder, oc.BuiltinCode())
        tflite.OperatorCode.OperatorCodeAddVersion(builder, oc.Version())
        if custom_off:
            tflite.OperatorCode.OperatorCodeAddCustomCode(builder, custom_off)
        opcode_offsets.append(tflite.OperatorCode.OperatorCodeEnd(builder))

    # --- SUBGRAPHS ---
    subgraph_offsets = []
    # we will need FullyConnected enum values
    BUILTIN_FULLY = getattr(tflite, "BuiltinOperator_FULLY_CONNECTED", None)
    BUILTINOPT_FC = getattr(tflite, "BuiltinOptions_FullyConnectedOptions", None)

    for si in range(model.SubgraphsLength()):
        sg = model.Subgraphs(si)

        # TENSORS: create tensor offsets (create name & shape vectors before starting tensors)
        tensor_offsets = []
        for ti in range(sg.TensorsLength()):
            t = sg.Tensors(ti)
            name = t.Name().decode() if t.Name() else ""
            shape = [t.Shape(j) for j in range(t.ShapeLength())] if t.ShapeLength() else []
            tensor_offsets.append(create_tensor_offset(builder, name, shape, t.Type(), t.Buffer()))

        # OPERATORS: for each operator, create inputs/outputs vectors BEFORE starting Operator table
        op_offsets = []
        for oi in range(sg.OperatorsLength()):
            op = sg.Operators(oi)
            inputs = [op.Inputs(j) for j in range(op.InputsLength())] if op.InputsLength() else []
            outputs = [op.Outputs(j) for j in range(op.OutputsLength())] if op.OutputsLength() else []
            inp_vec = vec_int(builder, inputs) if inputs else 0
            out_vec = vec_int(builder, outputs) if outputs else 0

            tflite.Operator.OperatorStart(builder)
            tflite.Operator.OperatorAddOpcodeIndex(builder, op.OpcodeIndex())
            if inp_vec:
                tflite.Operator.OperatorAddInputs(builder, inp_vec)
            if out_vec:
                tflite.Operator.OperatorAddOutputs(builder, out_vec)

            # If this operator is FULLY_CONNECTED, build FullyConnectedOptions BEFORE ending operator
            opcode_idx = op.OpcodeIndex()
            builtin_code = model.OperatorCodes(opcode_idx).BuiltinCode() if opcode_idx < model.OperatorCodesLength() else None
            if builtin_code == BUILTIN_FULLY:
                # build options (these are simple: no vectors inside)
                tflite.FullyConnectedOptions.FullyConnectedOptionsStart(builder)
                tflite.FullyConnectedOptions.FullyConnectedOptionsAddKeepNumDims(builder, 1)
                fc_off = tflite.FullyConnectedOptions.FullyConnectedOptionsEnd(builder)
                tflite.Operator.OperatorAddBuiltinOptions(builder, fc_off)
                if BUILTINOPT_FC is not None:
                    tflite.Operator.OperatorAddBuiltinOptionsType(builder, BUILTINOPT_FC)
            # else: leave options absent (copying arbitrary existing options exactly is more involved)

            op_offsets.append(tflite.Operator.OperatorEnd(builder))

        # now create vectors and subgraph table (all offsets already created)
        # tensors vector
        builder.StartVector(4, len(tensor_offsets), 4)
        for to in reversed(tensor_offsets):
            builder.PrependUOffsetTRelative(to)
        tensors_vec = builder.EndVector()

        # inputs vector
        inputs_idx = [sg.Inputs(i) for i in range(sg.InputsLength())] if sg.InputsLength() else []
        inputs_vec = vec_int(builder, inputs_idx) if inputs_idx else 0

        # outputs vector
        outputs_idx = [sg.Outputs(i) for i in range(sg.OutputsLength())] if sg.OutputsLength() else []
        outputs_vec = vec_int(builder, outputs_idx) if outputs_idx else 0

        # operators vector
        builder.StartVector(4, len(op_offsets), 4)
        for oo in reversed(op_offsets):
            builder.PrependUOffsetTRelative(oo)
        ops_vec = builder.EndVector()

        name_off = builder.CreateString(sg.Name().decode() if sg.Name() else "")

        tflite.SubGraph.SubGraphStart(builder)
        tflite.SubGraph.SubGraphAddTensors(builder, tensors_vec)
        if inputs_vec:
            tflite.SubGraph.SubGraphAddInputs(builder, inputs_vec)
        if outputs_vec:
            tflite.SubGraph.SubGraphAddOutputs(builder, outputs_vec)
        tflite.SubGraph.SubGraphAddOperators(builder, ops_vec)
        tflite.SubGraph.SubGraphAddName(builder, name_off)
        subgraph_offsets.append(tflite.SubGraph.SubGraphEnd(builder))

    # --- Model --- create top-level vectors (opcode, subgraphs, buffers) BEFORE ModelStart fields
    # operator codes vector
    builder.StartVector(4, len(opcode_offsets), 4)
    for oc in reversed(opcode_offsets):
        builder.PrependUOffsetTRelative(oc)
    opcodes_vec = builder.EndVector()

    # subgraphs vector
    builder.StartVector(4, len(subgraph_offsets), 4)
    for sg_off in reversed(subgraph_offsets):
        builder.PrependUOffsetTRelative(sg_off)
    subgraphs_vec = builder.EndVector()

    # buffers vector
    builder.StartVector(4, len(buffer_offsets), 4)
    for b_off in reversed(buffer_offsets):
        builder.PrependUOffsetTRelative(b_off)
    buffers_vec = builder.EndVector()

    # build model
    tflite.Model.ModelStart(builder)
    tflite.Model.ModelAddOperatorCodes(builder, opcodes_vec)
    tflite.Model.ModelAddSubgraphs(builder, subgraphs_vec)
    tflite.Model.ModelAddBuffers(builder, buffers_vec)
    tflite.Model.ModelAddDescription(builder, builder.CreateString("Injected keep_num_dims"))
    model_off = tflite.Model.ModelEnd(builder)
    builder.Finish(model_off)

    out = builder.Output()
    with open(output_path, "wb") as f:
        f.write(out)
    print("[OK] Saved:", output_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inject_keep_num_dims_fixed.py input.tflite output.tflite")
        sys.exit(1)
    inject = inject  # no-op to satisfy linters
    inject = None
    # call function
    inject = locals().get('inject') or globals().get('inject')  # ensure name resolution
    inject(sys.argv[1], sys.argv[2])

