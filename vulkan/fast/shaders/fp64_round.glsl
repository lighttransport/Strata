// Explicit IEEE binary64 -> binary32 round-to-nearest-even. Keeping the
// intermediate value as integer bits prevents a later FP64 operation from
// retaining precision across the required FP32 rounding boundary.
float round_fp64_to_fp32(double value) {
  const uvec2 bits = unpackDouble2x32(value);
  const uint sign = bits.y & 0x80000000u, exponent = (bits.y >> 20u) & 2047u;
  if (exponent == 2047u)
    return uintBitsToFloat(sign | (((bits.y & 0xfffffu) | bits.x) == 0u ? 0x7f800000u : 0x7fc00000u));
  if (exponent > 1150u) return uintBitsToFloat(sign | 0x7f800000u);
  if (exponent >= 897u) {
    uint result = sign | ((exponent - 896u) << 23u) | ((bits.y & 0xfffffu) << 3u) | (bits.x >> 29u);
    const uint lost = bits.x & 0x1fffffffu;
    if (lost > 0x10000000u || (lost == 0x10000000u && (result & 1u) != 0u)) ++result;
    return uintBitsToFloat(result);
  }
  if (exponent < 873u) return uintBitsToFloat(sign);
  const uint high = (bits.y & 0xfffffu) | 0x100000u, shift = 926u - exponent;
  uint result;
  bool increment;
  if (shift < 32u) {
    result = (high << (32u - shift)) | (bits.x >> shift);
    const uint lost = bits.x & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
    increment = lost > halfway || (lost == halfway && (result & 1u) != 0u);
  } else if (shift == 32u) {
    result = high;
    increment = bits.x > 0x80000000u || (bits.x == 0x80000000u && (result & 1u) != 0u);
  } else {
    const uint s = shift - 32u, lost = high & ((1u << s) - 1u), halfway = 1u << (s - 1u);
    result = high >> s;
    increment = lost > halfway || (lost == halfway && (bits.x != 0u || (result & 1u) != 0u));
  }
  return uintBitsToFloat(sign | (result + (increment ? 1u : 0u)));
}

bool fp64_at_fp32_midpoint(uvec2 bits) {
  const uint exponent = (bits.y >> 20u) & 2047u;
  if (exponent >= 897u && exponent <= 1150u) return (bits.x & 0x1fffffffu) == 0x10000000u;
  if (exponent < 873u || exponent > 896u) return false;
  const uint shift = 926u - exponent;
  if (shift < 32u) return (bits.x & ((1u << shift) - 1u)) == (1u << (shift - 1u));
  if (shift == 32u) return bits.x == 0x80000000u;
  const uint s = shift - 32u, high = (bits.y & 0xfffffu) | 0x100000u;
  return bits.x == 0u && (high & ((1u << s) - 1u)) == (1u << (s - 1u));
}

float fp32_fma_rne(float a, float b, float c) {
  // The FP32 product is exact in FP64. Recover a rounded FP64 sum's residual
  // only when it lands exactly on an FP32 midpoint; otherwise another FP32
  // rounding cannot cross a midpoint. This handles double-rounding cases.
  precise double product = double(a) * double(b);
  precise double value = product + double(c);
  uvec2 bits = unpackDouble2x32(value);
  if (fp64_at_fp32_midpoint(bits)) {
    precise double b_part = value - product;
    precise double a_part = value - b_part;
    precise double a_error = product - a_part;
    precise double b_error = double(c) - b_part;
    precise double error = a_error + b_error;
    if (error != 0.0) {
      const bool increase = ((bits.y & 0x80000000u) == 0u) == (error > 0.0);
      if (increase) { ++bits.x; if (bits.x == 0u) ++bits.y; }
      else { if (bits.x == 0u) --bits.y; --bits.x; }
    }
  }
  return round_fp64_to_fp32(packDouble2x32(bits));
}
