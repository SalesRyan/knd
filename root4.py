# dentro do loop sobre operadores, onde você tem 'op' (original) e 'op.OpcodeIndex()' etc.
# também precisa de 'data' (bytes do arquivo original) e 'model' (objeto root)

# detecta enums
BUILTIN_FULLY = getattr(tflite, "BuiltinOperator_FULLY_CONNECTED", None)
BUILTINOPT_FC = getattr(tflite, "BuiltinOptions_FullyConnectedOptions", None)

# ... já criou inp_vec/out_vec e fez OperatorStart + add opcode/inputs/outputs ...

opcode_idx = op.OpcodeIndex()
builtin_code_val = None
try:
    builtin_code_val = model.OperatorCodes(opcode_idx).BuiltinCode()
except Exception:
    builtin_code_val = None

if builtin_code_val == BUILTIN_FULLY:
    # verificar se o operador original tinha builtin options do tipo FullyConnectedOptions
    orig_btype = op.BuiltinOptionsType()
    fused_act = 0  # default NONE
    if orig_btype == BUILTINOPT_FC and op.BuiltinOptions() != 0:
        # ler o objeto FullyConnectedOptions do buffer original
        try:
            # Formas possíveis dependendo do binding gerado:
            # 1) tflite.FullyConnectedOptions.FullyConnectedOptions.GetRootAsFullyConnectedOptions
            # 2) tflite.FullyConnectedOptions.GetRootAsFullyConnectedOptions
            fc_mod = None
            if hasattr(tflite, "FullyConnectedOptions") and hasattr(tflite.FullyConnectedOptions, "FullyConnectedOptions"):
                fc_mod = tflite.FullyConnectedOptions.FullyConnectedOptions
            elif hasattr(tflite, "FullyConnectedOptions") and hasattr(tflite.FullyConnectedOptions, "GetRootAsFullyConnectedOptions"):
                # some bindings put functions directly under module
                fc_mod = tflite.FullyConnectedOptions
            if fc_mod is not None:
                # op.BuiltinOptions() é o offset dentro do flatbuffer; use-o como segundo arg
                fc_obj = None
                # duas convenções possíveis de getter:
                if hasattr(fc_mod, "GetRootAsFullyConnectedOptions"):
                    fc_obj = fc_mod.GetRootAsFullyConnectedOptions(data, op.BuiltinOptions())
                elif hasattr(fc_mod, "GetRootAs"):
                    fc_obj = fc_mod.GetRootAs(data, op.BuiltinOptions())
                else:
                    # fallback: tentar a classe diária
                    fc_obj = None
                if fc_obj is not None and hasattr(fc_obj, "FusedActivationFunction"):
                    fused_act = fc_obj.FusedActivationFunction()
        except Exception:
            fused_act = 0

    # Agora crie as opções preservando fused_act e setando keep_num_dims = 1
    tflite.FullyConnectedOptions.FullyConnectedOptionsStart(builder)
    # defina fused activation se o binding exige int; a função gerada geralmente espera int
    try:
        tflite.FullyConnectedOptions.FullyConnectedOptionsAddFusedActivationFunction(builder, int(fused_act))
    except Exception:
        # se não existir AddFusedActivationFunction (bindings diferentes), ignore e apenas set keep_num_dims
        pass
    # keep_num_dims
    tflite.FullyConnectedOptions.FullyConnectedOptionsAddKeepNumDims(builder, 1)
    fc_off = tflite.FullyConnectedOptions.FullyConnectedOptionsEnd(builder)

    tflite.Operator.OperatorAddBuiltinOptions(builder, fc_off)

    # setar builtin_options_type com o enum correto
    if BUILTINOPT_FC is not None:
        tflite.Operator.OperatorAddBuiltinOptionsType(builder, BUILTINOPT_FC)
    else:
        bo = getattr(tflite, "BuiltinOptions", None)
        if bo is not None and hasattr(bo, "FullyConnectedOptions"):
            tflite.Operator.OperatorAddBuiltinOptionsType(builder, bo.FullyConnectedOptions)