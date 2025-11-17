import flatbuffers
import tflite
import sys

def inject_keep_num_dims(input_path, output_path):
    data = open(input_path, "rb").read()
    model = tflite.Model.Model.GetRootAsModel(data, 0)

    builder = flatbuffers.Builder(2 * len(data))

    # helpers
    def vec_int(values):
        builder.StartVector(4, len(values), 4)
        for v in reversed(values):
            builder.PrependInt32(v)
        return builder.EndVector()

    # === Buffers ===
    buffers = []
    for i in range(model.BuffersLength()):
        buf = model.Buffers(i)
        raw = buf.DataAsNumpy() if hasattr(buf, "DataAsNumpy") else buf.Data()
        if raw is None:
            t = None
        else:
            try: t = bytes(raw)
            except: t = raw.tobytes()
        builder.StartVector(1, len(t) if t else 0, 1) if t else None
        if t:
            for b in reversed(t):
                builder.PrependUint8(b)
            data_vec = builder.EndVector()
        else:
            data_vec = 0
        tflite.Buffer.BufferStart(builder)
        if data_vec != 0:
            tflite.Buffer.BufferAddData(builder, data_vec)
        buffers.append(tflite.Buffer.BufferEnd(builder))

    # === OperatorCodes ===
    opcodes = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        tflite.OperatorCode.OperatorCodeStart(builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(builder, oc.BuiltinCode())
        tflite.OperatorCode.OperatorCodeAddVersion(builder, oc.Version())
        if oc.CustomCode():
            tflite.OperatorCode.OperatorCodeAddCustomCode(builder, builder.CreateString(oc.CustomCode().decode()))
        opcodes.append(tflite.OperatorCode.OperatorCodeEnd(builder))

    # === Subgraphs (tensors, operators, etc) ===
    subgraphs = []
    for si in range(model.SubgraphsLength()):
        sg = model.Subgraphs(si)

        # tensors
        tensors = []
        for ti in range(sg.TensorsLength()):
            t = sg.Tensors(ti)
            name = builder.CreateString(t.Name().decode() if t.Name() else "")
            shape = [t.Shape(j) for j in range(t.ShapeLength())]
            shape_vec = vec_int(shape) if shape else 0

            tflite.Tensor.TensorStart(builder)
            if shape_vec: tflite.Tensor.TensorAddShape(builder, shape_vec)
            tflite.Tensor.TensorAddType(builder, t.Type())
            tflite.Tensor.TensorAddBuffer(builder, t.Buffer())
            tflite.Tensor.TensorAddName(builder, name)
            tensors.append(tflite.Tensor.TensorEnd(builder))

        # operators
        ops = []
        for oi in range(sg.OperatorsLength()):
            op = sg.Operators(oi)

            inputs = [op.Inputs(j) for j in range(op.InputsLength())]
            outputs = [op.Outputs(j) for j in range(op.OutputsLength())]

            inp_vec = vec_int(inputs)
            out_vec = vec_int(outputs)

            tflite.Operator.OperatorStart(builder)
            tflite.Operator.OperatorAddOpcodeIndex(builder, op.OpcodeIndex())
            tflite.Operator.OperatorAddInputs(builder, inp_vec)
            tflite.Operator.OperatorAddOutputs(builder, out_vec)

            # ---- APPLY keep_num_dims = TRUE ----
            opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            if opcode == tflite.BuiltinOperator.BuiltinOperator().FULLY_CONNECTED:
                # build FullyConnectedOptions
                tflite.FullyConnectedOptions.FullyConnectedOptionsStart(builder)
                tflite.FullyConnectedOptions.FullyConnectedOptionsAddKeepNumDims(builder, True)
                opt = tflite.FullyConnectedOptions.FullyConnectedOptionsEnd(builder)

                tflite.Operator.OperatorAddBuiltinOptions(builder, opt)
                tflite.Operator.OperatorAddBuiltinOptionsType(
                    builder,
                    tflite.BuiltinOptions.BuiltinOptions().FullyConnectedOptions
                )

            ops.append(tflite.Operator.OperatorEnd(builder))

        # inputs / outputs
        in_vec  = vec_int([sg.Inputs(i)  for i in range(sg.InputsLength())])
        out_vec = vec_int([sg.Outputs(i) for i in range(sg.OutputsLength())])
        name = builder.CreateString(sg.Name().decode() if sg.Name() else "")

        tflite.SubGraph.SubGraphStart(builder)

        # tensors
        builder.StartVector(4, len(tensors), 4)
        for t in reversed(tensors):
            builder.PrependUOffsetTRelative(t)
        tflite.SubGraph.SubGraphAddTensors(builder, builder.EndVector())

        tflite.SubGraph.SubGraphAddInputs(builder, in_vec)
        tflite.SubGraph.SubGraphAddOutputs(builder, out_vec)
        tflite.SubGraph.SubGraphAddName(builder, name)

        # operators
        builder.StartVector(4, len(ops), 4)
        for o in reversed(ops):
            builder.PrependUOffsetTRelative(o)
        tflite.SubGraph.SubGraphAddOperators(builder, builder.EndVector())

        subgraphs.append(tflite.SubGraph.SubGraphEnd(builder))

    # === Model ===
    tflite.Model.ModelStart(builder)

    # operator codes
    builder.StartVector(4, len(opcodes), 4)
    for oc in reversed(opcodes):
        builder.PrependUOffsetTRelative(oc)
    tflite.Model.ModelAddOperatorCodes(builder, builder.EndVector())

    # subgraphs
    builder.StartVector(4, len(subgraphs), 4)
    for sg in reversed(subgraphs):
        builder.PrependUOffsetTRelative(sg)
    tflite.Model.ModelAddSubgraphs(builder, builder.EndVector())

    # buffers
    builder.StartVector(4, len(buffers), 4)
    for b in reversed(buffers):
        builder.PrependUOffsetTRelative(b)
    tflite.Model.ModelAddBuffers(builder, builder.EndVector())

    tflite.Model.ModelAddDescription(builder, builder.CreateString("Injected keep_num_dims"))
    model_off = tflite.Model.ModelEnd(builder)

    builder.Finish(model_off)
    open(output_path, "wb").write(builder.Output())
    print(f"[OK] Modelo salvo em: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python inject_keep_num_dims.py input.tflite output.tflite")
        sys.exit(1)
    inject_keep_num_dims(sys.argv[1], sys.argv[2])