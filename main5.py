import flatbuffers
import importlib
import sys
import tflite


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def vec_int(builder, values):
    if not values:
        return 0
    builder.StartVector(4, len(values), 4)
    for v in reversed(values):
        builder.PrependInt32(v)
    return builder.EndVector()


def create_string(builder, s):
    return builder.CreateString(s) if s else 0


def create_buffer(builder, raw_bytes):
    if not raw_bytes:
        tflite.Buffer.BufferStart(builder)
        return tflite.Buffer.BufferEnd(builder)
    off = builder.CreateByteVector(raw_bytes)
    tflite.Buffer.BufferStart(builder)
    tflite.Buffer.BufferAddData(builder, off)
    return tflite.Buffer.BufferEnd(builder)


def load_fc_options_from_original(data, op, fc_enum_type):
    """Lê as FullyConnectedOptions originais para preservar fused_activation_function."""
    if op.BuiltinOptionsType() != fc_enum_type:
        return 0  # NONE

    offset = op.BuiltinOptions()
    if offset == 0:
        return 0

    # Binding depende de versão do flatc → tentar formas
    try:
        FC = tflite.FullyConnectedOptions.FullyConnectedOptions
        obj = FC()
        obj.Init(data, offset)
        if hasattr(obj, "FusedActivationFunction"):
            return obj.FusedActivationFunction()
        return 0
    except Exception:
        return 0


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def inject_keepdims(input_path, output_path):
    data = open(input_path, "rb").read()

    # Carregar Model class
    try:
        ModelClass = importlib.import_module("tflite.Model").Model
    except Exception:
        ModelClass = tflite.Model.Model

    model = ModelClass.GetRootAsModel(data, 0)

    builder = flatbuffers.Builder(len(data) * 3)

    # Enums
    BUILTIN_FC = getattr(tflite, "BuiltinOperator_FULLY_CONNECTED", None)
    BUILTINOPTIONS_FC = getattr(tflite, "BuiltinOptions_FullyConnectedOptions", None)

    # ---------------------------------------------------------
    # Buffers
    # ---------------------------------------------------------
    orig_buffers = []
    for i in range(model.BuffersLength()):
        b = model.Buffers(i)
        if hasattr(b, "DataAsNumpy"):
            raw = b.DataAsNumpy()
        else:
            raw = b.Data()
        if raw is None:
            orig_buffers.append(None)
        else:
            try:
                orig_buffers.append(bytes(raw))
            except:
                orig_buffers.append(raw.tobytes())

    buffer_offs = [create_buffer(builder, rb) for rb in orig_buffers]

    # ---------------------------------------------------------
    # OperatorCodes
    # ---------------------------------------------------------
    opcode_offs = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        cc = oc.CustomCode().decode() if oc.CustomCode() else None
        cc_off = create_string(builder, cc)
        tflite.OperatorCode.OperatorCodeStart(builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(builder, oc.BuiltinCode())
        tflite.OperatorCode.OperatorCodeAddVersion(builder, oc.Version())
        if cc_off:
            tflite.OperatorCode.OperatorCodeAddCustomCode(builder, cc_off)
        opcode_offs.append(tflite.OperatorCode.OperatorCodeEnd(builder))

    # ---------------------------------------------------------
    # Subgraphs
    # ---------------------------------------------------------
    subgraph_offs = []
    for sg_i in range(model.SubgraphsLength()):
        sg = model.Subgraphs(sg_i)

        # Tensors
        tensor_offs = []
        for ti in range(sg.TensorsLength()):
            t = sg.Tensors(ti)
            name = t.Name().decode() if t.Name() else ""
            shape = [t.Shape(j) for j in range(t.ShapeLength())]

            name_off = create_string(builder, name)
            shape_off = vec_int(builder, shape)

            tflite.Tensor.TensorStart(builder)
            if shape_off:
                tflite.Tensor.TensorAddShape(builder, shape_off)
            tflite.Tensor.TensorAddType(builder, t.Type())
            tflite.Tensor.TensorAddBuffer(builder, t.Buffer())
            if name_off:
                tflite.Tensor.TensorAddName(builder, name_off)
            tensor_offs.append(tflite.Tensor.TensorEnd(builder))

        # Operators
        op_offs = []
        for oi in range(sg.OperatorsLength()):
            op = sg.Operators(oi)

            inputs = [op.Inputs(j) for j in range(op.InputsLength())]
            outputs = [op.Outputs(j) for j in range(op.OutputsLength())]

            in_vec = vec_int(builder, inputs)
            out_vec = vec_int(builder, outputs)

            opcode_idx = op.OpcodeIndex()
            builtin = None
            try:
                builtin = model.OperatorCodes(opcode_idx).BuiltinCode()
            except:
                builtin = None

            tflite.Operator.OperatorStart(builder)
            tflite.Operator.OperatorAddOpcodeIndex(builder, opcode_idx)
            if in_vec:
                tflite.Operator.OperatorAddInputs(builder, in_vec)
            if out_vec:
                tflite.Operator.OperatorAddOutputs(builder, out_vec)

            # ---------------------------------------------------------
            # Se for FULLY_CONNECTED → copiar options e injetar keep_num_dims
            # ---------------------------------------------------------
            if builtin == BUILTIN_FC:
                fused = load_fc_options_from_original(data, op, BUILTINOPTIONS_FC)

                tflite.FullyConnectedOptions.FullyConnectedOptionsStart(builder)

                # Preservar fused_activation_function
                if hasattr(tflite.FullyConnectedOptions, "FullyConnectedOptionsAddFusedActivationFunction"):
                    tflite.FullyConnectedOptions.FullyConnectedOptionsAddFusedActivationFunction(builder, int(fused))

                # Injetar keep_num_dims
                tflite.FullyConnectedOptions.FullyConnectedOptionsAddKeepNumDims(builder, 1)

                fc_off = tflite.FullyConnectedOptions.FullyConnectedOptionsEnd(builder)

                tflite.Operator.OperatorAddBuiltinOptions(builder, fc_off)

                if BUILTINOPTIONS_FC is not None:
                    tflite.Operator.OperatorAddBuiltinOptionsType(builder, BUILTINOPTIONS_FC)
                else:
                    # fallback
                    BO = getattr(tflite, "BuiltinOptions", None)
                    if BO and hasattr(BO, "FullyConnectedOptions"):
                        tflite.Operator.OperatorAddBuiltinOptionsType(builder, BO.FullyConnectedOptions)

            op_offs.append(tflite.Operator.OperatorEnd(builder))

        # Criar vetores finais
        builder.StartVector(4, len(tensor_offs), 4)
        for x in reversed(tensor_offs):
            builder.PrependUOffsetTRelative(x)
        tensors_vec = builder.EndVector()

        builder.StartVector(4, len(op_offs), 4)
        for x in reversed(op_offs):
            builder.PrependUOffsetTRelative(x)
        ops_vec = builder.EndVector()

        in_graph = vec_int(builder, [sg.Inputs(i) for i in range(sg.InputsLength())])
        out_graph = vec_int(builder, [sg.Outputs(i) for i in range(sg.OutputsLength())])
        name_sg = create_string(builder, sg.Name().decode() if sg.Name() else "")

        tflite.SubGraph.SubGraphStart(builder)
        tflite.SubGraph.SubGraphAddTensors(builder, tensors_vec)
        if in_graph:
            tflite.SubGraph.SubGraphAddInputs(builder, in_graph)
        if out_graph:
            tflite.SubGraph.SubGraphAddOutputs(builder, out_graph)
        tflite.SubGraph.SubGraphAddOperators(builder, ops_vec)
        if name_sg:
            tflite.SubGraph.SubGraphAddName(builder, name_sg)
        subgraph_offs.append(tflite.SubGraph.SubGraphEnd(builder))

    # ---------------------------------------------------------
    # Final Model
    # ---------------------------------------------------------
    builder.StartVector(4, len(opcode_offs), 4)
    for x in reversed(opcode_offs):
        builder.PrependUOffsetTRelative(x)
    opcodes_vec = builder.EndVector()

    builder.StartVector(4, len(subgraph_offs), 4)
    for x in reversed(subgraph_offs):
        builder.PrependUOffsetTRelative(x)
    subgraphs_vec = builder.EndVector()

    builder.StartVector(4, len(buffer_offs), 4)
    for x in reversed(buffer_offs):
        builder.PrependUOffsetTRelative(x)
    buffers_vec = builder.EndVector()

    tflite.Model.ModelStart(builder)
    tflite.Model.ModelAddVersion(builder, 3)
    tflite.Model.ModelAddOperatorCodes(builder, opcodes_vec)
    tflite.Model.ModelAddSubgraphs(builder, subgraphs_vec)
    tflite.Model.ModelAddBuffers(builder, buffers_vec)
    tflite.Model.ModelAddDescription(builder, builder.CreateString("Injected keep_num_dims"))
    model_off = tflite.Model.ModelEnd(builder)

    builder.Finish(model_off, b"TFL3")

    with open(output_path, "wb") as f:
        f.write(builder.Output())

    print("✔ Modelo salvo:", output_path)
    print("✔ keep_num_dims = TRUE aplicado")
    print("✔ fused_activation_function preservada")
    print("✔ Netron e NNX devem exibir corretamente")


# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python inject_keep_num_dims_full.py input.tflite output.tflite")
        sys.exit(1)

    inject_keepdims(sys.argv[1], sys.argv[2])
