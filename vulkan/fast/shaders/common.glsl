// Shared declarations for the fast GLM engine shaders.
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
