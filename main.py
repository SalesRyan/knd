# inject_keep_num_dims.py
# Uso: python inject_keep_num_dims.py input.tflite output.tflite
# Substitui/cria FullyConnectedOptions.keep_num_dims=1 para todos os FULLY_CONNECTED.

import sys
import flatbuffers
import tflite
import struct

def get(fn_name):
    return getattr(tflite, fn_name, None)

def read_file(path):
    with open(path, "rb") as f:
        return f.read()

def main(in_path, out_path):
    data = read_file(in_path)
    # obter raiz do modelo (binding gerado costuma expor Model.Model.GetRootAsModel)
    ModelGetRoot = None
    for cand in ("Model.Model.GetRootAsModel", "Model.GetRootAsModel"):
        parts = cand.split(".")
        try:
            # dynamic import chain
            obj = tflite
            for p in parts:
                obj = getattr(obj, p) if isinstance(obj, type) or hasattr(obj, p) else None
        except Exception:
            obj = None
        if obj:
            ModelGetRoot = obj
            break
    # fallback: tentar acesso direto comum
    if ModelGetRoot is None:
        try:
            ModelGetRoot = tflite.Model.Model.GetRootAsModel
        except Exception:
            pass
    if ModelGetRoot is None:
        print("Não encontrei Model.GetRootAsModel no módulo tflite. Rode:")
        print("print([n for n in dir(tflite) if 'Model' in n or 'GetRoot' in n])")
        raise SystemExit(1)

    model = ModelGetRoot(data, 0)

    # helpers para enums / valores
    BuiltinOperator_FULLY_CONNECTED = getattr(tflite, "BuiltinOperator_FULLY_CONNECTED", None)
    BuiltinOptions_FullyConnectedOptions = getattr(tflite, "BuiltinOptions_FullyConnectedOptions", None)
    if BuiltinOperator_FULLY_CONNECTED is None or BuiltinOptions_FullyConnectedOptions is None:
        print("Não encontrei enums BuiltinOperator_FULLY_CONNECTED ou BuiltinOptions_FullyConnectedOptions no módulo tflite.")
        raise SystemExit(1)

    # coletar operator codes (copiar como estão)
    opcodes = []
    n_opcodes = model.OperatorCodesLength()
    for i in range(n_opcodes):
        oc = model.OperatorCodes(i)
        # cada OperatorCode tem: BuiltinCode(), Version(), maybe CustomCode()
        opcodes.append({
            "builtin_code": oc.BuiltinCode(),
            "version": oc.Version(),
            "custom_code": oc.CustomCode().decode('utf-8') if oc.CustomCode() else None
        })

    # coletar buffers (copiar dados brutos)
    buffers = []
    for i in range(model.BuffersLength()):
        buf = model.Buffers(i)
        b = buf.DataAsNumpy() if hasattr(buf, "DataAsNumpy") else (buf.Data() if hasattr(buf, "Data") else None)
        # DataAsNumpy retorna numpy array or None; normalize to bytes or None
        if b is None:
            buffers.append(None)
        else:
            # b may be memoryview/bytes/numpy => convert to bytes
            try:
                buffers.append(bytes(b))
            except Exception:
                # fallback iterate
                buffers.append(b.tobytes() if hasattr(b, "tobytes") else bytes(b))

    # coletar subgraphs
    subgraphs = []
    for sg_i in range(model.SubgraphsLength()):
        sg = model.Subgraphs(sg_i)
        # tensors
        tensors = []
        for t_i in range(sg.TensorsLength()):
            t = sg.Tensors(t_i)
            # name decode safely
            name = t.Name().decode('utf-8') if t.Name() else ""
            # shape vector
            shape = [t.Shape(j) for j in range(t.ShapeLength())]
            tensors.append({
                "name": name,
                "shape": shape,
                "type": t.Type(),
                "buffer": t.Buffer(),
                "quantization": None  # ignored for simplicity
            })
        # operators
        operators = []
        for op_i in range(sg.OperatorsLength()):
            op = sg.Operators(op_i)
            # read inputs
            inputs = [op.Inputs(j) for j in range(op.InputsLength())]
            outputs = [op.Outputs(j) for j in range(op.OutputsLength())]
            opcode_index = op.OpcodeIndex()
            # builtins options type & raw bytes if present
            btype = op.BuiltinOptionsType()
            b_off = op.BuiltinOptions()
            # we can't directly extract raw flatbuffer of builtin options easily via accessor;
            # we'll rebuild everything, injecting new options when opcode is FULLY_CONNECTED
            operators.append({
                "inputs": inputs,
                "outputs": outputs,
                "opcode_index": opcode_index,
                "builtin_options_type": btype,
                "builtin_options_offset": b_off
            })
        # inputs/outputs indices
        sg_inputs = [sg.Inputs(j) for j in range(sg.InputsLength())]
        sg_outputs = [sg.Outputs(j) for j in range(sg.OutputsLength())]
        subgraphs.append({
            "name": sg.Name().decode('utf-8') if sg.Name() else "",
            "tensors": tensors,
            "inputs": sg_inputs,
            "outputs": sg_outputs,
            "operators": operators
        })

    # Agora reconstruir o modelo com um novo Builder, copiando tudo mas injetando FullyConnectedOptions.keep_num_dims=1
    builder = flatbuffers.Builder(len(data) + 1024)

    # Funções geradas
    OperatorCodeStart = get("OperatorCodeStart"); OperatorCodeAddBuiltinCode = get("OperatorCodeAddBuiltinCode")
    OperatorCodeAddVersion = get("OperatorCodeAddVersion"); OperatorCodeAddCustomCode = get("OperatorCodeAddCustomCode")
    OperatorCodeEnd = get("OperatorCodeEnd")
    BufferStart = get("BufferStart"); BufferAddData = get("BufferAddData"); BufferEnd = get("BufferEnd")
    TensorStart = get("TensorStart"); TensorAddShape = get("TensorAddShape"); TensorAddType = get("TensorAddType")
    TensorAddBuffer = get("TensorAddBuffer"); TensorAddName = get("TensorAddName"); TensorEnd = get("TensorEnd")
    SubGraphStart = get("SubGraphStart"); SubGraphAddTensors = get("SubGraphAddTensors")
    SubGraphAddInputs = get("SubGraphAddInputs"); SubGraphAddOutputs = get("SubGraphAddOutputs")
    SubGraphAddName = get("SubGraphAddName"); SubGraphAddOperators = get("SubGraphAddOperators"); SubGraphEnd = get("SubGraphEnd")
    OperatorStart = get("OperatorStart"); OperatorAddOpcodeIndex = get("OperatorAddOpcodeIndex")
    OperatorAddInputs = get("OperatorAddInputs"); OperatorAddOutputs = get("OperatorAddOutputs")
    OperatorAddBuiltinOptions = get("OperatorAddBuiltinOptions"); OperatorAddBuiltinOptionsType = get("OperatorAddBuiltinOptionsType")
    OperatorEnd = get("OperatorEnd")
    ModelStart = get("ModelStart"); ModelAddOperatorCodes = get("ModelAddOperatorCodes"); ModelAddSubgraphs = get("ModelAddSubgraphs")
    ModelAddBuffers = get("ModelAddBuffers"); ModelAddDescription = get("ModelAddDescription"); ModelEnd = get("ModelEnd")

    # sanity checks
    required = [OperatorCodeStart, OperatorCodeAddBuiltinCode, OperatorCodeEnd, BufferStart, BufferEnd,
                TensorStart, TensorEnd, SubGraphStart, SubGraphEnd, OperatorStart, OperatorEnd,
                OperatorAddBuiltinOptions, OperatorAddBuiltinOptionsType, ModelStart, ModelEnd]
    if any(fn is None for fn in required):
        print("Bindings esperados não encontrados no módulo tflite. Rode:")
        print("print([n for n in dir(tflite) if any(k in n for k in ['Operator', 'Model', 'SubGraph', 'Buffer', 'Tensor', 'FullyConnectedOptions'])])")
        raise SystemExit(1)

    # 1) criar buffers (buffer 0..N)
    buffer_offsets = []
    for b in buffers:
        BufferStart(builder)
        if b:
            data_vec = builder.CreateByteVector(b)
            BufferAddData(builder, data_vec)
        buf_off = BufferEnd(builder)
        buffer_offsets.append(buf_off)

    # 2) criar tensors (por subgraph) - manter nome strings
    # Para reconstruir subgraphs mais facilmente, vamos criar cada tensor e guardar seus offsets
    all_subgraph_tensor_offsets = []
    for sg in subgraphs:
        t_offsets = []
        for t in sg["tensors"]:
            name_off = builder.CreateString(t["name"])
            # shape vector
            if len(t["shape"]) > 0:
                builder.StartVector(4, len(t["shape"]), 4)
                for d in reversed(t["shape"]):
                    builder.PrependInt32(d)
                shape_vec = builder.EndVector()
            else:
                shape_vec = 0
            TensorStart(builder)
            if shape_vec:
                TensorAddShape(builder, shape_vec)
            TensorAddType(builder, t["type"])
            TensorAddBuffer(builder, t["buffer"])
            TensorAddName(builder, name_off)
            t_off = TensorEnd(builder)
            t_offsets.append(t_off)
        all_subgraph_tensor_offsets.append(t_offsets)

    # 3) criar OperatorCodes (copiar)
    opcodes_offsets = []
    for oc in opcodes:
        OperatorCodeStart(builder)
        OperatorCodeAddBuiltinCode(builder, oc["builtin_code"])
        if oc["version"] is not None:
            OperatorCodeAddVersion(builder, oc["version"])
        if oc["custom_code"]:
            OperatorCodeAddCustomCode(builder, builder.CreateString(oc["custom_code"]))
        op_off = OperatorCodeEnd(builder)
        opcodes_offsets.append(op_off)

    # 4) para cada subgraph, criar operators (injetando FullyConnectedOptions quando necessário) e depois o subgraph
    subgraph_offsets = []
    # prepare references to FullyConnectedOptions builder funcs
    fc_start = get("FullyConnectedOptionsStart"); fc_add_keep = get("FullyConnectedOptionsAddKeepNumDims"); fc_end = get("FullyConnectedOptionsEnd")
    for sidx, sg in enumerate(subgraphs):
        # criar operators vector
        op_offsets = []
        for op in sg["operators"]:
            # inputs vector
            builder.StartVector(4, len(op["inputs"]), 4)
            for i in reversed(op["inputs"]):
                builder.PrependInt32(i)
            inputs_vec = builder.EndVector()
            # outputs vector
            builder.StartVector(4, len(op["outputs"]), 4)
            for o in reversed(op["outputs"]):
                builder.PrependInt32(o)
            outputs_vec = builder.EndVector()

            OperatorStart(builder)
            OperatorAddOpcodeIndex(builder, op["opcode_index"])
            OperatorAddInputs(builder, inputs_vec)
            OperatorAddOutputs(builder, outputs_vec)

            # se opcode corresponde a FULLY_CONNECTED, criar/injetar FullyConnectedOptions with keep_num_dims=1
            opcode_builtin = opcodes[op["opcode_index"]]["builtin_code"] if op["opcode_index"] < len(opcodes) else None
            if opcode_builtin == BuiltinOperator_FULLY_CONNECTED:
                if not (fc_start and fc_add_keep and fc_end):
                    print("Funções FullyConnectedOptions não encontradas no binding.")
                    raise SystemExit(1)
                fc_start(builder)
                fc_add_keep(builder, 1)
                fc_off = fc_end(builder)
                OperatorAddBuiltinOptions(builder, fc_off)
                OperatorAddBuiltinOptionsType(builder, BuiltinOptions_FullyConnectedOptions)
            else:
                # sem opções: nada a fazer (copiá-las com fidelidade requer extra parsing; mantemos sem opções)
                pass

            op_off = OperatorEnd(builder)
            op_offsets.append(op_off)

        # criar vector de operators
        SubGraphStart(builder)
        # tensors vector para esta subgraph
        tvecs = all_subgraph_tensor_offsets[sidx]
        builder.StartVector(4, len(tvecs), 4)
        for tro in reversed(tvecs):
            builder.PrependUOffsetTRelative(tro)
        tensors_vec = builder.EndVector()
        SubGraphAddTensors(builder, tensors_vec)

        # inputs vector
        builder.StartVector(4, len(sg["inputs"]), 4)
        for ii in reversed(sg["inputs"]):
            builder.PrependInt32(ii)
        inputs_vec = builder.EndVector()
        SubGraphAddInputs(builder, inputs_vec)
        # outputs vector
        builder.StartVector(4, len(sg["outputs"]), 4)
        for oo in reversed(sg["outputs"]):
            builder.PrependInt32(oo)
        outputs_vec = builder.EndVector()
        SubGraphAddOutputs(builder, outputs_vec)

        SubGraphAddName(builder, builder.CreateString(sg["name"]))

        # operators vector
        builder.StartVector(4, len(op_offsets), 4)
        for opo in reversed(op_offsets):
            builder.PrependUOffsetTRelative(opo)
        ops_vec = builder.EndVector()
        SubGraphAddOperators(builder, ops_vec)

        sg_off = SubGraphEnd(builder)
        subgraph_offsets.append(sg_off)

    # 5) montar Model: operator codes, subgraphs, buffers
    ModelStart(builder)
    # operator codes vector
    builder.StartVector(4, len(opcodes_offsets), 4)
    for oc in reversed(opcodes_offsets):
        builder.PrependUOffsetTRelative(oc)
    ModelAddOperatorCodes(builder, builder.EndVector())
    # subgraphs vector
    builder.StartVector(4, len(subgraph_offsets), 4)
    for so in reversed(subgraph_offsets):
        builder.PrependUOffsetTRelative(so)
    ModelAddSubgraphs(builder, builder.EndVector())
    # buffers vector
    builder.StartVector(4, len(buffer_offsets), 4)
    for bo in reversed(buffer_offsets):
        builder.PrependUOffsetTRelative(bo)
    ModelAddBuffers(builder, builder.EndVector())
    ModelAddDescription(builder, builder.CreateString("Injected keep_num_dims=1 into FullyConnected ops"))
    model_off = ModelEnd(builder)

    builder.Finish(model_off)
    out_bytes = builder.Output()

    with open(out_path, "wb") as f:
        f.write(out_bytes)

    print(f"Wrote new model to {out_path} — ALL FullyConnected ops now have keep_num_dims=1 (injected).")

if _name_ == "_main_":
    if len(sys.argv) < 3:
        print("Uso: python inject_keep_num_dims.py input.tflite output.tflite")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])