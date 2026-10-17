#extension GL_EXT_spirv_intrinsics : require
spirv_instruction(extensions = ["SPV_KHR_fma"], capabilities = [6030], id = 4427)
float native_fp32_fma(float a, float b, float c);

// Experimental FP32 division with one residual correction. This is not a
// promise of correctly rounded division for all IEEE inputs. Keep its error
// and whole-model quality checks separate from the exact FP64 mode.
float corrected_fp32_div(float a, float b) {
  precise float q = a / b;
  if (isinf(q) || isnan(q) || isinf(b) || b == 0.0) return q;
  float residual = native_fp32_fma(-q, b, a);
  precise float correction = residual / b;
  precise float result = q + correction;
  return result;
}
