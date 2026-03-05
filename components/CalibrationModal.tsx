"use client";

import { useState } from "react";
import { X, RotateCcw, SlidersHorizontal } from "lucide-react";
import { useRos, CalibrationData } from "@/context/RosContext";

const DEFAULT_CALIBRATION: CalibrationData = {
  offsets: [0, 0, 0, 0, 0, 0, 0, 0],
  flips: [false, false, false, false, false, false, false, false],
  tcpOffset: { x: 0, y: 0, z: 0 },
};

const AXES = [
  { label: "J1", unit: "°", color: "bg-gray-100 text-gray-700" },
  { label: "J2", unit: "°", color: "bg-gray-100 text-gray-700" },
  { label: "J3", unit: "°", color: "bg-gray-100 text-gray-700" },
  { label: "J4", unit: "°", color: "bg-gray-100 text-gray-700" },
  { label: "J5", unit: "°", color: "bg-gray-100 text-gray-700" },
  { label: "J6", unit: "°", color: "bg-gray-100 text-gray-700" },
  { label: "Rail", unit: "mm", color: "bg-blue-100 text-blue-700" },
  { label: "Gripper", unit: "%", color: "bg-orange-100 text-orange-700" },
];

const TCP_AXES = ["x", "y", "z"] as const;

export default function CalibrationModal({ onClose }: { onClose: () => void }) {
  const { calibration, setCalibration } = useRos();
  const [draft, setDraft] = useState<CalibrationData>({
    offsets: [...calibration.offsets],
    flips: [...calibration.flips],
    tcpOffset: { ...calibration.tcpOffset },
  });

  const setOffset = (i: number, value: number) => {
    const next = [...draft.offsets];
    next[i] = Math.round(value * 10) / 10;
    setDraft({ ...draft, offsets: next });
  };

  const setFlip = (i: number, value: boolean) => {
    const next = [...draft.flips];
    next[i] = value;
    setDraft({ ...draft, flips: next });
  };

  const setTcp = (axis: "x" | "y" | "z", value: number) => {
    setDraft({ ...draft, tcpOffset: { ...draft.tcpOffset, [axis]: Math.round(value * 10) / 10 } });
  };

  const handleApply = () => {
    setCalibration(draft);
    onClose();
  };

  const handleReset = () => {
    setDraft({
      offsets: [...DEFAULT_CALIBRATION.offsets],
      flips: [...DEFAULT_CALIBRATION.flips],
      tcpOffset: { x: 0, y: 0, z: 0 },
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6">
      <div className="tesla-card w-full max-w-xl flex flex-col max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-8 pt-8 pb-6 border-b border-black/5">
          <div className="flex items-center gap-3">
            <SlidersHorizontal size={22} className="text-gray-500" />
            <h2 className="text-2xl font-black">Calibration</h2>
          </div>
          <button
            onClick={onClose}
            className="w-10 h-10 rounded-full bg-gray-100 hover:bg-gray-200 flex items-center justify-center transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        <div className="overflow-y-auto flex-1 px-8 py-4 space-y-6">
          {/* Axis rows */}
          <div className="space-y-1">
            <div className="grid grid-cols-[80px_1fr_160px] gap-4 py-2">
              <span className="text-xs font-black text-gray-400 uppercase">Axis</span>
              <span className="text-xs font-black text-gray-400 uppercase text-center">Flip (กลับทิศ)</span>
              <span className="text-xs font-black text-gray-400 uppercase text-center">Offset</span>
            </div>
            {AXES.map((axis, i) => (
              <div key={i} className="grid grid-cols-[80px_1fr_160px] gap-4 items-center py-1">
                <span className={`inline-flex items-center justify-center w-16 h-9 rounded-full text-sm font-black ${axis.color}`}>
                  {axis.label}
                </span>
                <div className="flex justify-center">
                  <button
                    onClick={() => setFlip(i, !draft.flips[i])}
                    className={`relative w-14 h-8 rounded-full transition-colors ${draft.flips[i] ? "bg-blue-500" : "bg-gray-200"}`}
                  >
                    <div className={`absolute top-1 w-6 h-6 rounded-full bg-white shadow-md transition-all duration-200 ${draft.flips[i] ? "left-7" : "left-1"}`} />
                  </button>
                </div>
                <div className="flex items-center gap-1">
                  <button onClick={() => setOffset(i, draft.offsets[i] - 1)} className="w-10 h-10 rounded-2xl bg-gray-100 hover:bg-gray-200 text-lg font-black flex items-center justify-center transition-colors">−</button>
                  <input
                    type="number"
                    value={draft.offsets[i]}
                    onChange={(e) => setOffset(i, parseFloat(e.target.value) || 0)}
                    className="w-20 h-10 text-center font-mono font-black text-base rounded-2xl border-2 border-gray-200 bg-white focus:outline-none focus:border-blue-400"
                  />
                  <button onClick={() => setOffset(i, draft.offsets[i] + 1)} className="w-10 h-10 rounded-2xl bg-gray-100 hover:bg-gray-200 text-lg font-black flex items-center justify-center transition-colors">+</button>
                </div>
              </div>
            ))}
          </div>

          {/* TCP Offset section */}
          <div className="border-t border-gray-100 pt-5">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-xl bg-purple-100 flex items-center justify-center">
                <span className="text-purple-600 text-xs font-black">TCP</span>
              </div>
              <div>
                <p className="text-sm font-black text-gray-700">Tool Center Point Offset</p>
                <p className="text-xs text-gray-400">ระยะห่างจากปลาย Joint 6 ถึงปลายเครื่องมือ (mm)</p>
              </div>
            </div>
            <div className="space-y-2">
              {TCP_AXES.map((axis) => (
                <div key={axis} className="flex items-center gap-3">
                  <span className="w-14 text-center inline-flex items-center justify-center h-9 rounded-full text-sm font-black bg-purple-100 text-purple-700 uppercase">{axis}</span>
                  <div className="flex items-center gap-1">
                    <button onClick={() => setTcp(axis, draft.tcpOffset[axis] - 1)} className="w-10 h-10 rounded-2xl bg-gray-100 hover:bg-gray-200 text-lg font-black flex items-center justify-center transition-colors">−</button>
                    <input
                      type="number"
                      value={draft.tcpOffset[axis]}
                      onChange={(e) => setTcp(axis, parseFloat(e.target.value) || 0)}
                      className="w-20 h-10 text-center font-mono font-black text-base rounded-2xl border-2 border-gray-200 bg-white focus:outline-none focus:border-purple-400"
                    />
                    <button onClick={() => setTcp(axis, draft.tcpOffset[axis] + 1)} className="w-10 h-10 rounded-2xl bg-gray-100 hover:bg-gray-200 text-lg font-black flex items-center justify-center transition-colors">+</button>
                  </div>
                  <span className="text-xs text-gray-400 font-bold">mm</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center gap-3 px-8 pt-4 pb-8 border-t border-black/5">
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-6 py-3 rounded-2xl bg-gray-100 hover:bg-gray-200 text-gray-600 font-black transition-colors"
          >
            <RotateCcw size={16} /> Reset All
          </button>
          <button
            onClick={handleApply}
            className="flex-1 py-3 rounded-2xl bg-black hover:bg-gray-800 active:bg-gray-900 text-white font-black text-base transition-colors"
          >
            Apply
          </button>
        </div>
      </div>
    </div>
  );
}
