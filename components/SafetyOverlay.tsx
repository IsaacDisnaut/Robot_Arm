"use client";

import { useRos } from "@/context/RosContext";
import { AlertTriangle, ShieldAlert } from "lucide-react";

export default function SafetyOverlay() {
  const { safetyStatus } = useRos();

  if (safetyStatus === 0) return null;

  const isEmergency = safetyStatus === 2;

  return (
    <>
      {/* ── Inset border frame around entire viewport ─────────────────── */}
      {/* pointer-events-none: never blocks taps/clicks */}
      <div
        className={`fixed inset-0 pointer-events-none z-[9990] ${
          isEmergency ? "safety-frame-emergency" : "safety-frame-warn"
        }`}
      />

      {/* ── Emergency: top full-width strip ───────────────────────────── */}
      {isEmergency && (
        <div className="fixed top-0 left-0 right-0 z-[9995] safety-strip-flash pointer-events-none">
          <div className="bg-red-600 text-white flex items-center justify-center gap-3 py-3 px-6 font-black text-base tracking-wide">
            <ShieldAlert size={20} />
            หยุดฉุกเฉิน! — มีคนอยู่ในพื้นที่อันตราย
            <ShieldAlert size={20} />
          </div>
        </div>
      )}

      {/* ── Bottom floating pill ──────────────────────────────────────── */}
      {/* Positioned above typical footer buttons (bottom-[90px]) */}
      <div
        className={`fixed bottom-[90px] left-1/2 -translate-x-1/2 z-[9995] pointer-events-none
          flex items-center gap-2.5 px-6 py-3.5 rounded-full font-black text-white text-sm
          shadow-2xl whitespace-nowrap
          ${
            isEmergency
              ? "bg-red-600 shadow-red-600/50 safety-strip-flash"
              : "bg-orange-500 shadow-orange-500/40"
          }`}
      >
        <AlertTriangle size={17} />
        {isEmergency
          ? "🛑 หยุดฉุกเฉิน! — มีคนอยู่ใกล้หุ่น"
          : "⚠ คำเตือน — มีคนเข้าใกล้หุ่นยนต์ กำลังลดความเร็ว"}
        <AlertTriangle size={17} />
      </div>

      {/* ── Emergency: dark edge vignette (extra drama) ──────────────── */}
      {isEmergency && (
        <div
          className="fixed inset-0 pointer-events-none z-[9989]"
          style={{
            background:
              "radial-gradient(ellipse at center, transparent 55%, rgba(220,38,38,0.22) 100%)",
          }}
        />
      )}
    </>
  );
}
