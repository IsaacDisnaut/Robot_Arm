"use client";

import { useRos } from "@/context/RosContext";
import { ShieldCheck } from "lucide-react";

export default function RosStatusBadge() {
  const { isConnected, safetyStatus, isTestMode, setTestMode } = useRos();

  const safety =
    safetyStatus === 2
      ? { label: "ฉุกเฉิน!", bg: "bg-red-500 animate-pulse", text: "text-white" }
      : safetyStatus === 1
        ? { label: "เตือน", bg: "bg-orange-500", text: "text-white" }
        : { label: "ปกติ", bg: "bg-emerald-500", text: "text-white" };

  return (
    <div className="flex items-center gap-2">
      {/* ROS connection */}
      <div
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold ${
          isConnected
            ? "bg-green-50 text-green-700"
            : "bg-red-50 text-red-600"
        }`}
      >
        <div
          className={`w-1.5 h-1.5 rounded-full ${
            isConnected ? "bg-green-500" : "bg-red-500 animate-pulse"
          }`}
        />
        {isConnected ? "ROS" : "ROS Offline"}
      </div>

      {/* SIM mode toggle */}
      <button
        onClick={() => setTestMode(!isTestMode)}
        className={`px-3 py-1.5 rounded-full text-xs font-bold transition-colors ${
          isTestMode
            ? "bg-orange-500 text-white"
            : "bg-gray-100 text-gray-400 hover:bg-gray-200"
        }`}
        title={isTestMode ? "กำลังใช้ Simulation Mode — กดเพื่อปิด" : "กดเพื่อเปิด Simulation Mode"}
      >
        SIM
      </button>

      {/* Safety status — big pill */}
      <div
        className={`flex items-center gap-2 px-5 py-2 rounded-full font-black text-sm ${safety.bg} ${safety.text}`}
      >
        <ShieldCheck size={15} />
        {safety.label}
      </div>
    </div>
  );
}
