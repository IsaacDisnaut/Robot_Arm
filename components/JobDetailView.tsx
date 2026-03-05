"use client";

import React, { useState, useEffect, useRef, useMemo } from "react";
import { useRos } from "@/context/RosContext";
import RosStatusBadge from "@/components/RosStatusBadge";
import JobEditor from "@/components/JobEditor";
import {
  ArrowLeft,
  Play,
  Edit2,
  Trash2,
  X,
  Clock,
  Zap,
  Pause,
  Square,
  Activity,
  RotateCcw,
  MoveHorizontal,
  Hand,
  Home,
  FileJson,
  Copy,
  Check,
  Download,
} from "lucide-react";

interface Task {
  id: number;
  sequence: number;
  label?: string;
  j1: number;
  j2: number;
  j3: number;
  j4: number;
  j5: number;
  j6: number;
  rail: number;
  speed?: number;
  delay?: number;
  gripper?: number;
  controlMode?: string;
  x?: number | null;
  y?: number | null;
  z?: number | null;
  roll?: number | null;
  pitch?: number | null;
  yaw?: number | null;
}

interface Job {
  id: number;
  name: string;
  description?: string;
  tasks?: Task[];
  createdAt?: string;
}

interface JobDetailViewProps {
  job: Job;
  onBack: () => void;
  onUpdate: () => void;
  autoStart?: boolean;
  autoHomeOnComplete?: boolean;
}

// ─── helpers ────────────────────────────────────────────────
const formatTime = (ms: number) => {
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
};

// Estimate how long a task takes based on actual joint movement distance.
// Base assumption: at 100% speed, moving 180° takes ~3 seconds.
// Rail: 600mm full range at 100% speed ~3 seconds.
const estimateTaskTime = (task: Task, prevTask?: Task): number => {
  const speed = Math.max(1, task.speed || 50) / 100;
  const prev = prevTask ?? { j1: 0, j2: 0, j3: 0, j4: 0, j5: 0, j6: 0, rail: 0 };

  const jointDeltas = [
    Math.abs(task.j1 - prev.j1),
    Math.abs(task.j2 - prev.j2),
    Math.abs(task.j3 - prev.j3),
    Math.abs(task.j4 - prev.j4),
    Math.abs(task.j5 - prev.j5),
    Math.abs(task.j6 - prev.j6),
  ];
  const maxJointDelta = Math.max(...jointDeltas);
  const railDelta = Math.abs(task.rail - prev.rail);

  // movement time = proportional to largest motion / speed
  const jointTime = (maxJointDelta / 180) * 3000 / speed;
  const railTime  = (railDelta / 600) * 3000 / speed;
  const moveTime  = Math.max(jointTime, railTime, 500 / speed); // minimum 500ms at 100%

  return Math.round(moveTime) + (task.delay || 0);
};

// taskDurationMs wraps estimateTaskTime so the loop wait and progress bar always use
// the same formula. prevTask is passed so short moves animate short, long moves long.
const taskDurationMs = (task: Task, prevTask?: Task): number =>
  estimateTaskTime(task, prevTask);

// Position-based progress: use leading joint (max travel distance) as reference.
// Returns 0–1. If no joint/rail needs to move, returns 1 (already done).
const calcTaskProgress = (
  task: Task,
  prevTask: Task | undefined,
  joints: number[],
  rail: number
): number => {
  const prev = prevTask ?? { j1: 0, j2: 0, j3: 0, j4: 0, j5: 0, j6: 0, rail: 0 };
  const targets  = [task.j1, task.j2, task.j3, task.j4, task.j5, task.j6, task.rail];
  const starts   = [prev.j1, prev.j2, prev.j3, prev.j4, prev.j5, prev.j6, prev.rail];
  const currents = [...joints.slice(0, 6), rail];

  let maxDelta = 0;
  let leadProgress = 1;
  for (let i = 0; i < 7; i++) {
    const totalDelta = Math.abs(targets[i] - starts[i]);
    if (totalDelta > maxDelta) {
      maxDelta = totalDelta;
      leadProgress = Math.min(1, Math.abs(currents[i] - starts[i]) / totalDelta);
    }
  }
  return maxDelta < 0.5 ? 1 : leadProgress;
};

// ─── component ──────────────────────────────────────────────
export default function JobDetailView({ job, onBack, onUpdate, autoStart = false, autoHomeOnComplete = false }: JobDetailViewProps) {
  const {
    isConnected,
    robotStatus,
    stopExecution,
    pauseExecution,
    resumeExecution,
    sendGotoPosition,
    isTestMode,
    jointStates,
    jointVelocities,
    railPos,
    gripperPos,
    effectorPose,
  } = useRos();

  // Local execution state — fully decoupled from RosContext so robot feedback
  // never overrides what the user explicitly requested.
  const [isExecuting, setIsExecuting] = useState(false);
  const [localTaskIdx, setLocalTaskIdx] = useState(0);
  const [localIsPaused, setLocalIsPaused] = useState(false);

  const [isEditing, setIsEditing] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [taskElapsedMs, setTaskElapsedMs] = useState(0);
  const [showJson, setShowJson] = useState(false);
  const [copied, setCopied] = useState(false);

  const [showCompletionModal, setShowCompletionModal] = useState(false);
  const [totalElapsedMs, setTotalElapsedMs] = useState(0);
  const [delayCountdownMs, setDelayCountdownMs] = useState<number | null>(null);
  const delayTotalMsRef = useRef(0);
  const [startCountdown, setStartCountdown] = useState<number | null>(null);

  const taskRefs = useRef<(HTMLDivElement | null)[]>([]);
  const lastScrolledRef = useRef(-1);
  const executionCancelRef = useRef(false);
  const userPausedRef = useRef(false); // user's explicit pause/resume — NOT synced from robot
  const robotStatusRef = useRef(0);
  const robotWasMovingRef = useRef(false);
  const executionStartMsRef = useRef(0);
  // Keep latest startExecutionFlow in ref to avoid stale closure in countdown effect
  const startExecutionFlowRef = useRef<() => void>(() => {});

  // robot_status from ROS updates robotStatusRef; localIsPaused + userPausedRef are fully user-controlled
  useEffect(() => {
    robotStatusRef.current = robotStatus;
    if (isExecuting && robotStatus !== 0) robotWasMovingRef.current = true;
  }, [robotStatus, isExecuting]);

  const tasks = job.tasks || [];

  // ── per-task timer: resets when task changes, stops when paused
  useEffect(() => {
    setTaskElapsedMs(0);
    robotWasMovingRef.current = false; // reset per task so snap logic works fresh
  }, [localTaskIdx]);

  useEffect(() => {
    if (!isExecuting || localIsPaused) return;
    const interval = setInterval(() => setTaskElapsedMs((p) => p + 200), 200);
    return () => clearInterval(interval);
  }, [isExecuting, localIsPaused, localTaskIdx]);

  // ── reset on execution stop
  useEffect(() => {
    if (!isExecuting) {
      setTaskElapsedMs(0);
      lastScrolledRef.current = -1;
    }
  }, [isExecuting]);

  // ── auto-start countdown on mount
  useEffect(() => {
    if (autoStart && tasks.length > 0) {
      setStartCountdown(3);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── countdown tick → trigger execution at 0
  useEffect(() => {
    if (startCountdown === null) return;
    if (startCountdown === 0) {
      setStartCountdown(null);
      startExecutionFlowRef.current();
      return;
    }
    const timer = setTimeout(() => setStartCountdown((c) => (c !== null ? c - 1 : null)), 1000);
    return () => clearTimeout(timer);
  }, [startCountdown]);

  // ── derived values (use local state — not RosContext — for execution tracking)
  const currentIdx = isExecuting ? localTaskIdx : -1;
  const curTask = tasks[currentIdx >= 0 ? currentIdx : 0];
  // Movement time only (delay excluded — delay has its own countdown UI)
  const curDurationMs = curTask
    ? taskDurationMs(curTask, tasks[currentIdx - 1]) - (curTask.delay || 0)
    : 1;

  // Position-based progress (primary): use actual joint positions from ROS.
  // Fallback to time estimate in test mode or when no movement is detected.
  const posProgress = (isConnected && !isTestMode && curTask)
    ? calcTaskProgress(curTask, tasks[currentIdx - 1], jointStates, railPos)
    : null;
  // Snap to 100% the moment robot reports idle after having been busy
  const snapped = robotStatus === 0 && robotWasMovingRef.current;
  const curTaskPct = snapped
    ? 1
    : posProgress !== null
      ? posProgress
      : Math.min(1, taskElapsedMs / curDurationMs);
  const overallPct = tasks.length > 0 && isExecuting
    ? Math.round(((currentIdx + curTaskPct) / tasks.length) * 100)
    : 0;
  const remainingMs = isExecuting
    ? Math.max(0, curDurationMs - taskElapsedMs) +
      tasks.slice(currentIdx + 1).reduce((s, t, i) => s + taskDurationMs(t, tasks[currentIdx + i]), 0)
    : 0;

  // ── auto-scroll to current task
  useEffect(() => {
    if (!isExecuting || currentIdx < 0 || currentIdx === lastScrolledRef.current) return;
    lastScrolledRef.current = currentIdx;
    taskRefs.current[currentIdx]?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [currentIdx, isExecuting]);

  // ── JSON preview payload (same structure sent to /execute_trajectory)
  const rosPayload = useMemo(() => ({
    id: job.id,
    name: job.name,
    tasks: tasks.map((t) => ({
      sequence: t.sequence,
      label: t.label ?? `Task ${t.sequence}`,
      j1: t.j1, j2: t.j2, j3: t.j3,
      j4: t.j4, j5: t.j5, j6: t.j6,
      rail: t.rail,
      speed: t.speed ?? 50,
      delay: t.delay ?? 0,
      gripper: t.gripper ?? 0,
    })),
  }), [job.id, job.name, tasks]);

  const copyJson = async () => {
    await navigator.clipboard.writeText(JSON.stringify(rosPayload, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadJson = () => {
    const now = new Date();
    const ts = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}_${String(now.getHours()).padStart(2, "0")}${String(now.getMinutes()).padStart(2, "0")}${String(now.getSeconds()).padStart(2, "0")}`;
    const safeName = job.name.replace(/[^a-zA-Z0-9ก-๙\s-]/g, "").trim().replace(/\s+/g, "_");
    const filename = `${safeName}_${ts}.json`;
    const blob = new Blob([JSON.stringify(rosPayload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── delete job
  const deleteJob = async () => {
    if (!confirm(`Delete job "${job.name}"? This cannot be undone.`)) return;
    setIsDeleting(true);
    try {
      const res = await fetch(`/api/jobs/${job.id}`, { method: "DELETE" });
      if (res.ok) onBack();
    } catch (e) {
      console.error(e);
    } finally {
      setIsDeleting(false);
    }
  };

  // ── wait for robot movement. Returns true if interrupted by Pause (caller should retry task).
  const waitForRobotMovement = async (task: Task, prevTask?: Task): Promise<boolean> => {
    const moveMsEst = estimateTaskTime(task, prevTask) - (task.delay || 0);
    let interrupted = false;

    // Wait up to 600ms for robot to start (status becomes non-zero)
    const startDeadline = Date.now() + 600;
    while (robotStatusRef.current === 0 && Date.now() < startDeadline) {
      if (executionCancelRef.current) return false;
      if (userPausedRef.current) break; // paused before robot started
      await new Promise(r => setTimeout(r, 50));
    }

    if (robotStatusRef.current !== 0) {
      // Robot is moving — wait for idle; send stop immediately if user pauses
      let moveDone = Date.now() + moveMsEst * 3 + 3000;
      while (robotStatusRef.current !== 0 && Date.now() < moveDone) {
        if (executionCancelRef.current) return false;
        if (userPausedRef.current) {
          if (!interrupted) {
            stopExecution(); // halt robot NOW (publishes /stop_execution)
            interrupted = true;
          }
          moveDone += 50; // wait for robot to confirm stop (status=0)
        }
        await new Promise(r => setTimeout(r, 50));
      }
    } else {
      // No /robot_status feedback — use exact same formula as Dry Run
      const waitMs = Math.max(2000, (100 - (task.speed ?? 50)) / 100 * 5000);
      let elapsed = 0;
      while (elapsed < waitMs) {
        if (executionCancelRef.current) return false;
        if (userPausedRef.current) {
          stopExecution(); // halt robot
          interrupted = true;
          break;
        }
        await new Promise(r => setTimeout(r, 50));
        elapsed += 50;
      }
    }

    // Hold here until user resumes
    while (userPausedRef.current && !executionCancelRef.current) {
      await new Promise(r => setTimeout(r, 100));
    }
    return interrupted;
  };

  // ── countdown delay with live popup (only called when delay > 0)
  const runDelayCountdown = async (delayMs: number): Promise<void> => {
    delayTotalMsRef.current = delayMs;
    setDelayCountdownMs(delayMs);
    let remaining = delayMs;
    while (remaining > 0) {
      if (executionCancelRef.current) break;
      await new Promise(r => setTimeout(r, 100));
      if (!userPausedRef.current) remaining -= 100;
      setDelayCountdownMs(Math.max(0, remaining));
    }
    setDelayCountdownMs(null);
  };

  // ── pause/resume: update local state + ref before sending to ROS
  const handlePause = () => {
    userPausedRef.current = true;
    setLocalIsPaused(true);
    pauseExecution();
  };

  const handleResume = () => {
    userPausedRef.current = false;
    setLocalIsPaused(false);
    resumeExecution();
  };

  // ── start execution: send one task at a time via /goto_position (same as Dry Run)
  const startExecutionFlow = async () => {
    if (!isTestMode && !isConnected) {
      alert("กรุณาเชื่อมต่อ ROS ก่อนเริ่มงาน\nหรือกด SIM ในแถบสถานะเพื่อทดสอบ");
      return;
    }
    if (tasks.length === 0) return;

    executionCancelRef.current = false;
    userPausedRef.current = false;
    setLocalIsPaused(false);
    executionStartMsRef.current = Date.now();
    setIsExecuting(true);
    setLocalTaskIdx(0);

    let i = 0;
    while (i < tasks.length) {
      if (executionCancelRef.current) break;

      setLocalTaskIdx(i);
      const task = tasks[i];

      // Don't send next command while paused between tasks
      while (userPausedRef.current && !executionCancelRef.current) {
        await new Promise(r => setTimeout(r, 100));
      }
      if (executionCancelRef.current) break;

      if (!isTestMode) {
        sendGotoPosition({
          sequence: task.sequence,
          label: task.label ?? `Task ${task.sequence}`,
          j1: task.j1, j2: task.j2, j3: task.j3,
          j4: task.j4, j5: task.j5, j6: task.j6,
          rail: task.rail,
          speed: task.speed ?? 50,
          gripper: task.gripper ?? 0,
        });
      }

      const interrupted = await waitForRobotMovement(task, tasks[i - 1]);
      if (interrupted && !executionCancelRef.current) {
        // Robot was halted mid-task — retry same task (robot moves from stopped position to target)
        continue;
      }

      const effectiveDelay = task.delay ?? 0;
      if (effectiveDelay > 0 && !executionCancelRef.current) {
        await runDelayCountdown(effectiveDelay);
      }
      i++;
    }

    // Execution complete (either all tasks done or cancelled)
    const elapsed = Date.now() - executionStartMsRef.current;
    setIsExecuting(false);
    setLocalTaskIdx(0);
    if (executionCancelRef.current) {
      stopExecution(); // send /stop_execution only when cancelled
    } else {
      // All tasks completed normally — show popup then auto-return
      setTotalElapsedMs(elapsed);
      setShowCompletionModal(true);
      if (autoHomeOnComplete) {
        sendGotoPosition({ sequence: 0, label: "Home", j1: 0, j2: 0, j3: 0, j4: 0, j5: 0, j6: 0, rail: 0, speed: 20, gripper: 0 });
      }
      setTimeout(() => {
        setShowCompletionModal(false);
        onBack();
      }, 3000);
    }
  };

  // Keep ref in sync with latest version of the function (avoids stale closure in countdown)
  startExecutionFlowRef.current = startExecutionFlow;

  // ── stop: cancel async loop + send stop signal to ROS
  const handleStop = () => {
    executionCancelRef.current = true;
    userPausedRef.current = false;
    setLocalIsPaused(false);
    setIsExecuting(false);
    setLocalTaskIdx(0);
    stopExecution();
  };

  if (isEditing) {
    return (
      <JobEditor
        mode="edit"
        job={job}
        onSave={() => { setIsEditing(false); onUpdate(); }}
        onCancel={() => setIsEditing(false)}
      />
    );
  }

  return (
    <div className="h-screen flex flex-col bg-[#F5F5F7]">

      {/* ── Header ─────────────────────────────────────────── */}
      <div className="bg-white border-b border-gray-100 px-5 pt-4 pb-3 shrink-0">
        {/* Row 1: back + title + actions */}
        <div className="flex items-center justify-between gap-4">

          {/* Left: Back button + title */}
          <div className="flex items-center gap-4 min-w-0">
            <button
              onClick={onBack}
              disabled={isExecuting}
              className="flex items-center gap-2 px-5 py-3 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-2xl transition-colors shrink-0 disabled:opacity-30 disabled:cursor-not-allowed font-bold text-gray-700"
            >
              <ArrowLeft size={20} />
              <span>กลับ</span>
            </button>
            <div className="min-w-0">
              <h1 className="text-xl font-black leading-tight truncate">{job.name}</h1>
              <p className="text-gray-400 text-xs mt-0.5 truncate">
                {job.description || "No description"}
              </p>
            </div>
          </div>

          {/* Right: status badges + action buttons */}
          <div className="flex items-center gap-2 shrink-0">
            {isTestMode && (
              <span className="px-3 py-1.5 bg-orange-500 text-white text-xs font-black rounded-full">
                🧪 DEMO
              </span>
            )}
            <RosStatusBadge />

            {!isExecuting && (
              <>
                {/* Secondary actions — smaller, less prominent */}
                <div className="w-px h-8 bg-gray-200 mx-1" />
                <button
                  onClick={() => setShowJson(true)}
                  className="p-3 bg-gray-100 text-gray-500 hover:text-purple-600 hover:bg-purple-50 rounded-xl transition-colors"
                  title="ดู JSON ที่ส่งไป ROS"
                >
                  <FileJson size={18} />
                </button>
                <button
                  onClick={deleteJob}
                  disabled={isDeleting}
                  className="p-3 bg-gray-100 text-gray-500 hover:text-red-500 hover:bg-red-50 rounded-xl transition-colors disabled:opacity-40"
                >
                  <Trash2 size={18} />
                </button>

                {/* Primary action — Edit */}
                <div className="w-px h-8 bg-gray-200 mx-1" />
                <button
                  onClick={() => setIsEditing(true)}
                  className="flex items-center gap-2 px-5 py-3 bg-blue-50 text-blue-600 hover:bg-blue-100 active:bg-blue-200 rounded-2xl transition-colors font-bold"
                >
                  <Edit2 size={18} />
                  <span>แก้ไข</span>
                </button>
              </>
            )}

            {/* Home — always visible, large touch target */}
            <button
              onClick={() => sendGotoPosition({ sequence: 0, label: "Home", j1: 0, j2: 0, j3: 0, j4: 0, j5: 0, j6: 0, rail: 0, speed: 20, gripper: 0 })}
              disabled={isExecuting}
              className="flex items-center gap-2 px-5 py-3 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 text-gray-700 rounded-2xl transition-colors disabled:opacity-30 font-bold"
              title="ส่งหุ่นยนต์กลับตำแหน่ง Home"
            >
              <Home size={18} />
              <span>Home</span>
            </button>
          </div>
        </div>

        {/* Row 2: stats */}
        <div className="flex items-center gap-5 mt-3 pl-1 text-sm text-gray-400 font-medium">
          <span className="flex items-center gap-1.5">
            <Activity size={14} />
            {tasks.length} tasks
          </span>
          <span className="flex items-center gap-1.5">
            <Clock size={14} />
            {formatTime(tasks.reduce((s, t, i) => s + taskDurationMs(t, tasks[i - 1]), 0))}
          </span>
          <span className="flex items-center gap-1.5">
            <Zap size={14} />
            {tasks.length > 0
              ? Math.round(tasks.reduce((s, t) => s + (t.speed || 50), 0) / tasks.length)
              : 0}% avg
          </span>
        </div>
      </div>

      {/* ── Body: left real-time panel + right task panel ── */}
      <div className="flex-1 flex overflow-hidden">

      {/* ── Left panel: real-time data ───────────────────── */}
      <div className="w-56 bg-white border-r border-gray-100 shrink-0 flex flex-col p-4 gap-3 overflow-y-auto">
        <p className="text-[10px] font-black text-gray-400 uppercase flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-blue-500 animate-ping inline-block" />
          Real-time
        </p>

        {/* Joints — compact 3-column */}
        <div className="grid grid-cols-3 gap-1.5">
          {jointStates.map((v, i) => (
            <div key={i} className="p-2 bg-gray-50 rounded-xl text-center">
              <span className="text-[8px] font-black text-gray-400 block uppercase">J{i + 1}</span>
              <span className="text-xs font-mono font-black">{v.toFixed(1)}°</span>
              <span className="text-[8px] text-gray-300 font-mono block">{(jointVelocities[i] ?? 0).toFixed(0)}°/s</span>
            </div>
          ))}
        </div>

        {/* Effector Pose */}
        <div className="p-3 bg-purple-50 rounded-2xl">
          <p className="text-[8px] font-black text-purple-400 uppercase mb-1.5">Effector Pose</p>
          <div className="grid grid-cols-3 gap-x-2 gap-y-0.5 text-[10px] font-mono text-purple-600">
            <div><span className="opacity-50">X </span>{effectorPose.x.toFixed(0)}</div>
            <div><span className="opacity-50">Y </span>{effectorPose.y.toFixed(0)}</div>
            <div><span className="opacity-50">Z </span>{effectorPose.z.toFixed(0)}</div>
            <div><span className="opacity-50">R </span>{effectorPose.roll.toFixed(1)}°</div>
            <div><span className="opacity-50">P </span>{effectorPose.pitch.toFixed(1)}°</div>
            <div><span className="opacity-50">Yw </span>{effectorPose.yaw.toFixed(1)}°</div>
          </div>
        </div>

        {/* Rail */}
        <div className="p-3 bg-blue-600 rounded-2xl text-white">
          <p className="text-[8px] font-black opacity-60 uppercase mb-0.5">Rail</p>
          <p className="font-mono font-black text-base">
            {railPos.toFixed(1)}<span className="text-[10px] font-light opacity-50 ml-1">mm</span>
          </p>
        </div>

        {/* Gripper */}
        <div className="p-3 bg-orange-50 rounded-2xl">
          <p className="text-[8px] font-black text-orange-400 uppercase mb-0.5">Gripper</p>
          <p className="font-mono font-black text-base text-orange-500">
            {gripperPos.toFixed(0)}<span className="text-[10px] font-light opacity-50 ml-1">%</span>
          </p>
          <div className="mt-1.5 w-full h-1 bg-orange-100 rounded-full overflow-hidden">
            <div className="h-full bg-orange-400 rounded-full transition-all duration-300" style={{ width: `${gripperPos}%` }} />
          </div>
        </div>
      </div>

      {/* ── Right panel: execution controls + task list ───── */}
      <div className="flex-1 flex flex-col overflow-hidden">

      {/* ── Execution control bar ────────────────────────── */}
      {isExecuting && (
        <div className="bg-white border-b border-gray-100 px-6 py-4 shrink-0">
          {/* Status + progress */}
          <div className="flex items-center gap-4 mb-3">
            <div className="flex-1">
              <div className="flex justify-between items-center mb-1.5">
                <span className="text-xs font-black text-gray-400 uppercase flex items-center gap-2">
                  {localIsPaused ? (
                    <span className="text-orange-500">⏸ Paused</span>
                  ) : (
                    <>
                      <span className="w-2 h-2 rounded-full bg-blue-500 animate-ping inline-block" />
                      Task {currentIdx + 1} / {tasks.length}
                    </>
                  )}
                </span>
                <span className="text-sm font-black text-blue-600">{overallPct}%</span>
              </div>
              <div className="w-full h-3 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    localIsPaused
                      ? "bg-orange-400"
                      : "bg-gradient-to-r from-blue-500 to-indigo-500"
                  }`}
                  style={{ width: `${overallPct}%` }}
                />
              </div>
            </div>
            <div className="text-right shrink-0">
              <p className="text-[10px] text-gray-400 font-bold uppercase">remaining</p>
              <p className="text-base font-black text-gray-700">{formatTime(remainingMs)}</p>
            </div>
          </div>

          {/* Pause / Resume / Stop */}
          <div className="flex gap-3">
            {localIsPaused ? (
              <button
                onClick={handleResume}
                className="flex-1 flex items-center justify-center gap-2 py-3 rounded-2xl bg-blue-600 text-white font-bold hover:bg-blue-700 active:bg-blue-800 transition-colors"
              >
                <Play size={18} fill="white" /> Resume
              </button>
            ) : (
              <button
                onClick={handlePause}
                className="flex-1 flex items-center justify-center gap-2 py-3 rounded-2xl bg-gray-100 text-gray-700 font-bold hover:bg-gray-200 active:bg-gray-300 transition-colors"
              >
                <Pause size={18} /> Pause
              </button>
            )}
            <button
              onClick={handleStop}
              className="flex-1 flex items-center justify-center gap-2 py-3 rounded-2xl bg-red-50 text-red-600 font-bold hover:bg-red-100 active:bg-red-200 transition-colors"
            >
              <Square size={18} fill="currentColor" /> Stop
            </button>
          </div>
        </div>
      )}

      {/* ── Task list ────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-5 space-y-3">
        {tasks.map((task, idx) => {
          const isActive = isExecuting && idx === currentIdx;
          const isDone = isExecuting && idx < currentIdx;
          const pct = isActive
            ? Math.round(curTaskPct * 100)
            : isDone ? 100 : 0;
          const estMs = estimateTaskTime(task, idx > 0 ? tasks[idx - 1] : undefined);

          return (
            <div
              key={task.id}
              ref={(el) => { taskRefs.current[idx] = el; }}
              className={`rounded-2xl border-2 p-5 transition-all duration-300 ${
                isActive
                  ? "border-blue-500 bg-blue-50 shadow-lg shadow-blue-100/60"
                  : isDone
                    ? "border-green-400 bg-green-50"
                    : "border-gray-200 bg-white"
              }`}
            >
              {/* Top row: badge + label + meta */}
              <div className="flex items-start gap-4 mb-3">
                {/* Sequence badge */}
                <div
                  className={`w-11 h-11 rounded-2xl flex items-center justify-center font-black text-white text-base shrink-0 transition-all ${
                    isActive
                      ? "bg-blue-600 scale-110"
                      : isDone
                        ? "bg-green-500"
                        : "bg-gray-300"
                  }`}
                >
                  {isActive ? (
                    <Play size={18} fill="white" className="animate-pulse" />
                  ) : isDone ? (
                    "✓"
                  ) : (
                    task.sequence
                  )}
                </div>

                {/* Label + metadata */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h3 className="font-black text-lg leading-tight">
                      {task.label || `Task ${task.sequence}`}
                    </h3>
                    {isActive && (
                      <span className="text-[11px] font-black text-blue-600 bg-blue-100 px-2 py-0.5 rounded-full animate-pulse">
                        ● IN PROGRESS
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 mt-1 flex-wrap">
                    <span className="flex items-center gap-1 text-xs text-gray-400">
                      <Clock size={11} /> {formatTime(estMs)}
                    </span>
                    <span className="flex items-center gap-1 text-xs text-gray-400">
                      <Zap size={11} /> {task.speed || 50}%
                    </span>
                    {(task.delay || 0) > 0 && (
                      <span className="text-xs text-gray-400">{task.delay}ms delay</span>
                    )}
                    {(task.gripper ?? 0) > 0 && (
                      <span className="text-xs font-bold bg-orange-100 text-orange-600 px-2 py-0.5 rounded-full">
                        ✊ {task.gripper}%
                      </span>
                    )}
                    <span className={`text-[10px] font-black px-2 py-0.5 rounded-full ${
                      (task.controlMode ?? "joint") === "effector"
                        ? "bg-purple-100 text-purple-700"
                        : "bg-gray-100 text-gray-500"
                    }`}>
                      {(task.controlMode ?? "joint") === "effector" ? "Effector" : "Joint"}
                    </span>
                  </div>
                </div>

                {/* Progress % */}
                {isExecuting && (
                  <span
                    className={`text-xl font-black shrink-0 tabular-nums ${
                      isActive ? "text-blue-600" : isDone ? "text-green-600" : "text-gray-300"
                    }`}
                  >
                    {pct}%
                  </span>
                )}
              </div>

              {/* Position values with icons */}
              <div
                className={`grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs mb-3 ${
                  isActive ? "text-blue-500" : isDone ? "text-green-600" : "text-gray-400"
                }`}
              >
                <div className="flex items-center gap-1.5 min-w-0">
                  <RotateCcw size={10} className="opacity-50 shrink-0" />
                  <span className="font-black opacity-50 w-8 shrink-0">J1-J3</span>
                  <span className="font-mono truncate">
                    {task.j1.toFixed(1)}°/{task.j2.toFixed(1)}°/{task.j3.toFixed(1)}°
                  </span>
                </div>
                <div className="flex items-center gap-1.5 min-w-0">
                  <RotateCcw size={10} className="opacity-50 shrink-0" />
                  <span className="font-black opacity-50 w-8 shrink-0">J4-J6</span>
                  <span className="font-mono truncate">
                    {task.j4.toFixed(1)}°/{task.j5.toFixed(1)}°/{task.j6.toFixed(1)}°
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <MoveHorizontal size={10} className="opacity-50 shrink-0" />
                  <span className="font-black opacity-50 w-8 shrink-0">Rail</span>
                  <span className="font-mono">{task.rail.toFixed(1)} mm</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Hand size={10} className="opacity-50 shrink-0" />
                  <span className="font-black opacity-50 w-8 shrink-0">Grip</span>
                  <span
                    className={`font-mono ${
                      (task.gripper ?? 0) > 0 ? "text-orange-500 font-bold" : ""
                    }`}
                  >
                    {task.gripper ?? 0}%
                  </span>
                </div>
                {task.x != null && (
                  <>
                    <div className="flex items-center gap-1.5 min-w-0 text-purple-500">
                      <span className="font-black opacity-60 w-8 shrink-0">XYZ</span>
                      <span className="font-mono truncate">
                        {task.x.toFixed(0)}/{task.y!.toFixed(0)}/{task.z!.toFixed(0)} mm
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5 min-w-0 text-purple-500">
                      <span className="font-black opacity-60 w-8 shrink-0">RPY</span>
                      <span className="font-mono truncate">
                        {task.roll!.toFixed(1)}°/{task.pitch!.toFixed(1)}°/{task.yaw!.toFixed(1)}°
                      </span>
                    </div>
                  </>
                )}
              </div>

              {/* Per-task progress bar */}
              <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-200 ${
                    isDone
                      ? "bg-green-500"
                      : isActive
                        ? localIsPaused
                          ? "bg-orange-400"
                          : "bg-blue-500"
                        : "bg-transparent"
                  }`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}

        {tasks.length === 0 && (
          <div className="text-center text-gray-400 py-16">
            <Activity size={40} className="mx-auto mb-4 opacity-30" />
            <p>No tasks in this job.</p>
          </div>
        )}
      </div>

      {/* ── Footer: Start button (only when not executing and countdown is not active) ── */}
      {!isExecuting && startCountdown === null && (
        <div className="bg-white border-t border-gray-100 p-5 shrink-0">
          <button
            onClick={startExecutionFlow}
            className="w-full py-4 bg-blue-600 text-white rounded-full font-black text-lg hover:bg-blue-700 active:bg-blue-800 transition-colors flex items-center justify-center gap-3 shadow-lg"
          >
            <Play size={22} fill="white" /> เริ่มทำงาน
          </button>
        </div>
      )}

      </div> {/* end right panel */}
      </div> {/* end body */}

      {/* ── Start Countdown Overlay ───────────────────────── */}
      {startCountdown !== null && startCountdown > 0 && (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/85 backdrop-blur-sm">
          <p className="text-white/60 text-lg font-bold mb-6 uppercase tracking-widest">กำลังเริ่ม</p>
          <p className="text-white font-black tabular-nums" style={{ fontSize: "200px", lineHeight: 1 }}>
            {startCountdown}
          </p>
          <p className="text-white/40 text-base mt-4">{job.name}</p>
          <button
            onClick={() => setStartCountdown(null)}
            className="mt-12 flex items-center gap-2 px-8 py-4 bg-white/15 hover:bg-white/25 text-white rounded-2xl font-bold text-lg transition-colors"
          >
            <X size={20} /> ยกเลิก
          </button>
        </div>
      )}

      {/* ── Delay Countdown ───────────────────────────────── */}
      {delayCountdownMs !== null && delayCountdownMs > 0 && (
        <div className="fixed inset-0 z-40 flex items-center justify-center pointer-events-none">
          <div className="bg-gray-900/95 backdrop-blur-md rounded-3xl px-16 py-12 shadow-2xl text-center min-w-[300px]">
            <p className="text-white/50 text-xs font-black uppercase tracking-widest mb-4">
              Task {localTaskIdx + 1} เสร็จแล้ว — รอก่อน
            </p>
            <p className="text-orange-400 font-black tabular-nums leading-none"
               style={{ fontSize: "80px" }}>
              {(delayCountdownMs / 1000).toFixed(1)}
            </p>
            <p className="text-white/40 text-sm mt-1">วินาที</p>
            <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden mt-6">
              <div
                className="h-full bg-orange-400 rounded-full transition-none"
                style={{ width: `${(delayCountdownMs / delayTotalMsRef.current) * 100}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* ── Completion Modal ──────────────────────────────── */}
      {showCompletionModal && (
        <div className="fixed inset-0 z-50 flex items-end justify-center pb-20 pointer-events-none">
          <div className="bg-white rounded-3xl px-10 py-8 shadow-2xl flex items-center gap-6 pointer-events-auto">
            <div className="w-14 h-14 rounded-full bg-green-500 flex items-center justify-center shrink-0">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </div>
            <div>
              <p className="text-xl font-black">เสร็จสิ้น!</p>
              <p className="text-gray-400 text-sm">
                ทำครบ {tasks.length} tasks · ใช้เวลา {formatTime(totalElapsedMs)}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── JSON Preview Modal ────────────────────────────── */}
      {showJson && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/50 backdrop-blur-sm"
          onClick={() => setShowJson(false)}
        >
          <div
            className="w-full max-w-2xl bg-[#1E1E2E] rounded-3xl overflow-hidden shadow-2xl flex flex-col max-h-[80vh]"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
              <div className="flex items-center gap-3">
                <FileJson size={18} className="text-purple-400" />
                <div>
                  <p className="text-white font-black text-sm">/execute_trajectory payload</p>
                  <p className="text-white/40 text-xs">std_msgs/String → JSON.stringify()</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={downloadJson}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/10 hover:bg-white/20 text-white text-xs font-bold transition-colors"
                  title="ดาวน์โหลด JSON"
                >
                  <Download size={13} /> Download
                </button>
                <button
                  onClick={copyJson}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/10 hover:bg-white/20 text-white text-xs font-bold transition-colors"
                >
                  {copied ? <Check size={13} className="text-green-400" /> : <Copy size={13} />}
                  {copied ? "Copied!" : "Copy"}
                </button>
                <button
                  onClick={() => setShowJson(false)}
                  className="p-1.5 rounded-xl bg-white/10 hover:bg-white/20 text-white transition-colors"
                >
                  <X size={16} />
                </button>
              </div>
            </div>

            {/* JSON content */}
            <pre className="flex-1 overflow-y-auto p-6 text-xs leading-relaxed font-mono text-green-300 whitespace-pre">
              {JSON.stringify(rosPayload, null, 2)}
            </pre>

            {/* Footer */}
            <div className="px-6 py-3 border-t border-white/10 flex items-center justify-between">
              <span className="text-white/30 text-xs">{tasks.length} tasks · topic: /execute_trajectory</span>
              <span className="text-white/30 text-xs">{JSON.stringify(rosPayload).length} bytes</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
