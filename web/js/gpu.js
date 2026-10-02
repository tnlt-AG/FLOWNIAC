// WebGPU device setup, shared by the app and the parity test.

export class GPUUnavailable extends Error {}

export async function initGPU() {
  if (!navigator.gpu) {
    throw new GPUUnavailable("This browser does not support WebGPU. Use a current Chrome or Edge, " +
                             "Safari 26 or newer, or Firefox 141 or newer (Windows).");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new GPUUnavailable("WebGPU is switched off or no suitable graphics adapter was found.");
  }
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (e) => console.error("WebGPU:", e.error.message));
  return { adapter, device };
}
