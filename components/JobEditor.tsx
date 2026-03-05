"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRos } from "@/context/RosContext";
import {
  ArrowLeft,
  Save,
  Info,
  Trash2,
  GripHorizontal,
  Crosshair,
  Check,
  Play,
  Square,
  Loader2,
  RotateCcw,
  MoveHorizontal,
  Hand,
  Zap,
  Clock,
  Home,
  FileJson,
  X,
  Copy,
  Download,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import RosStatusBadge from "@/components/RosStatusBadge";
import {
  DndContext,
  closestCenter,
  PointerSensor,
  TouchSensor,
  useSensor,
  useSensors,
  DragEndEvent,
  DragOverlay,
  DragStartEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  verticalListSortingStrategy,
  useSortable,
  arrayMove,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

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

interface JobEditorProps {
  mode: "create" | "edit";
  job?: {
    id: number;
    name: string;
    description?: string;
    tasks?: Task[];
  };
  onSave: () => void;
  onCancel: () => void;
}

// --- Sortable Task Card ---
function SortableTaskCard({
  task,
  idx,
  onLabelChange,
  onSpeedChange,
  onDelayChange,
  onRecapture,
  onTest,
  onDelete,
  onShowTaskJson,
  onControlModeChange,
  isRunning = false,
  isDone = false,
}: {
  task: Task;
  idx: number;
  onLabelChange: (idx: number, value: string) => void;
  onSpeedChange: (idx: number, value: number) => void;
  onDelayChange: (idx: number, value: number) => void;
  onRecapture: (idx: number) => void;
  onTest: (idx: number) => void;
  onDelete: (idx: number) => void;
  onShowTaskJson: (idx: number) => void;
  onControlModeChange: (id: number, mode: string) => void;
  isRunning?: boolean;
  isDone?: boolean;
}) {
  const [captured, setCaptured] = useState(false);
  const [testing, setTesting] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging: isSortableDragging,
  } = useSortable({ id: task.id });

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isSortableDragging ? 0.35 : 1,
  };

  // Auto-scroll when this task becomes the running one
  useEffect(() => {
    if (isRunning && scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [isRunning]);

  const handleRecapture = () => {
    onRecapture(idx);
    setCaptured(true);
    setTimeout(() => setCaptured(false), 1500);
  };

  const handleTest = () => {
    onTest(idx);
    setTesting(true);
    setTimeout(() => setTesting(false), 3000);
  };

  return (
    <div
      ref={(el) => { setNodeRef(el); scrollRef.current = el; }}
      style={style}
      className={`tesla-card flex overflow-hidden transition-all ${
        isRunning
          ? "ring-2 ring-blue-500"
          : isDone
            ? "ring-2 ring-green-400"
            : testing
              ? "ring-2 ring-orange-400"
              : captured
                ? "ring-2 ring-green-400"
                : ""
      }`}
    >
      {/* Left Drag Handle Strip */}
      <button
        {...attributes}
        {...listeners}
        className="w-14 flex-shrink-0 flex flex-col items-center justify-center gap-1 text-gray-300 hover:text-gray-500 active:text-gray-700 cursor-grab active:cursor-grabbing touch-none bg-gray-50/80 border-r border-gray-100 hover:bg-gray-100 transition-colors"
        tabIndex={-1}
        aria-label="Drag to reorder"
      >
        <GripHorizontal size={24} />
        <span className="text-[9px] font-black text-gray-300 tracking-widest">DRAG</span>
      </button>

      {/* Main Content */}
      <div
        className={`flex-1 p-5 space-y-4 transition-colors ${
          isRunning ? "bg-blue-50/40" : isDone ? "bg-green-50/40" : captured ? "bg-green-50/40" : testing ? "bg-orange-50/30" : ""
        }`}
      >
        {/* Row 1: Sequence badge + Label input + Delete */}
        <div className="flex items-center gap-3">
          <div className={`w-11 h-11 rounded-[12px] flex items-center justify-center font-black text-base flex-shrink-0 transition-all ${
            isRunning ? "bg-blue-500 text-white scale-110" : isDone ? "bg-green-500 text-white" : "bg-gray-100 text-gray-500"
          }`}>
            {isRunning ? <Play size={16} fill="white" className="animate-pulse" /> : isDone ? "✓" : idx + 1}
          </div>
          <input
            type="text"
            value={task.label || ""}
            onChange={(e) => onLabelChange(idx, e.target.value)}
            placeholder={`Task ${idx + 1}`}
            className="flex-1 font-bold text-lg text-[#1D1D1F] bg-transparent border-b-2 border-transparent focus:border-blue-400 focus:outline-none px-1 py-1 transition-colors min-w-0"
          />
          {isRunning && (
            <span className="text-[11px] font-black text-blue-600 bg-blue-100 px-2 py-0.5 rounded-full animate-pulse flex-shrink-0">
              ● RUNNING
            </span>
          )}
          <button
            onClick={() => onDelete(idx)}
            className="w-11 h-11 flex items-center justify-center text-gray-300 hover:text-red-500 hover:bg-red-50 rounded-xl transition-colors flex-shrink-0"
          >
            <Trash2 size={20} />
          </button>
        </div>

        {/* Row 2: Position values with icons */}
        <div
          className={`grid grid-cols-2 gap-x-4 gap-y-2.5 transition-colors ${
            captured ? "text-green-600" : "text-gray-500"
          }`}
        >
          <div className="flex items-center gap-2 min-w-0">
            <RotateCcw size={13} className="opacity-40 flex-shrink-0" />
            <span className="text-[11px] font-black font-sans opacity-50 w-9 flex-shrink-0">
              J1-J3
            </span>
            <span className="font-mono font-bold text-xs truncate">
              {task.j1.toFixed(1)}°/{task.j2.toFixed(1)}°/{task.j3.toFixed(1)}°
            </span>
          </div>
          <div className="flex items-center gap-2 min-w-0">
            <RotateCcw size={13} className="opacity-40 flex-shrink-0" />
            <span className="text-[11px] font-black font-sans opacity-50 w-9 flex-shrink-0">
              J4-J6
            </span>
            <span className="font-mono font-bold text-xs truncate">
              {task.j4.toFixed(1)}°/{task.j5.toFixed(1)}°/{task.j6.toFixed(1)}°
            </span>
          </div>
          <div className="flex items-center gap-2">
            <MoveHorizontal size={13} className="opacity-40 flex-shrink-0" />
            <span className="text-[11px] font-black font-sans opacity-50 w-9 flex-shrink-0">
              Rail
            </span>
            <span className="font-mono font-bold text-xs">
              {task.rail.toFixed(1)} mm
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Hand size={13} className="opacity-40 flex-shrink-0" />
            <span className="text-[11px] font-black font-sans opacity-50 w-9 flex-shrink-0">
              Grip
            </span>
            <span
              className={`font-mono font-bold text-xs ${
                (task.gripper ?? 0) > 0 ? "text-orange-500" : ""
              }`}
            >
              {task.gripper ?? 0}%
            </span>
          </div>
          {task.x != null && (
            <>
              <div className="flex items-center gap-2 min-w-0 text-purple-500">
                <span className="text-[11px] font-black opacity-60 w-9 flex-shrink-0">XYZ</span>
                <span className="font-mono font-bold text-xs truncate">
                  {task.x.toFixed(0)}/{task.y!.toFixed(0)}/{task.z!.toFixed(0)} mm
                </span>
              </div>
              <div className="flex items-center gap-2 min-w-0 text-purple-500">
                <span className="text-[11px] font-black opacity-60 w-9 flex-shrink-0">RPY</span>
                <span className="font-mono font-bold text-xs truncate">
                  {task.roll!.toFixed(1)}°/{task.pitch!.toFixed(1)}°/{task.yaw!.toFixed(1)}°
                </span>
              </div>
            </>
          )}
        </div>

        {/* Row 3: Speed slider */}
        <div className="flex items-center gap-3">
          <Zap size={13} className="opacity-40 flex-shrink-0 text-gray-500" />
          <span className="text-[11px] font-black font-sans opacity-50 w-9 flex-shrink-0">
            Speed
          </span>
          <input
            type="range"
            min="10"
            max="100"
            step="5"
            value={task.speed ?? 50}
            onChange={(e) => onSpeedChange(idx, Number(e.target.value))}
            className="flex-1 accent-blue-500 h-1.5 cursor-pointer"
          />
          <span className="text-xs font-black text-blue-600 w-9 text-right tabular-nums">
            {task.speed ?? 50}%
          </span>
        </div>

        {/* Row 4: Delay after task */}
        <div className="flex items-center gap-3">
          <Clock size={13} className="opacity-40 flex-shrink-0 text-gray-500" />
          <span className="text-[11px] font-black font-sans opacity-50 w-9 flex-shrink-0">
            Delay
          </span>
          <input
            type="range"
            min="0"
            max="10"
            step="0.5"
            value={(task.delay ?? 0) / 1000}
            onChange={(e) => onDelayChange(idx, Math.round(Number(e.target.value) * 1000))}
            className="flex-1 accent-orange-400 h-1.5 cursor-pointer"
          />
          <span className="text-xs font-black text-orange-500 w-9 text-right tabular-nums">
            {((task.delay ?? 0) / 1000).toFixed(1)}s
          </span>
        </div>

        {/* Row 5: Control Mode toggle */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-black font-sans opacity-50 w-16 flex-shrink-0">Mode</span>
          {["joint", "effector"].map((m) => (
            <button
              key={m}
              onClick={() => onControlModeChange(task.id, m)}
              className={`px-3 py-1.5 rounded-full text-xs font-black transition-colors ${
                (task.controlMode ?? "joint") === m
                  ? m === "joint" ? "bg-gray-800 text-white" : "bg-purple-600 text-white"
                  : "bg-gray-100 text-gray-400 hover:bg-gray-200"
              }`}
            >
              {m === "joint" ? "Joint" : "Effector"}
            </button>
          ))}
        </div>

        {/* Row 3: Action Buttons — large for touch */}
        <div className="flex gap-3 pt-1">
          <button
            onClick={handleRecapture}
            disabled={testing}
            title="จับตำแหน่งปัจจุบันจากหุ่นยนต์"
            className={`flex-1 flex items-center justify-center gap-2.5 py-4 rounded-2xl font-bold text-base transition-all disabled:opacity-40 ${
              captured
                ? "bg-green-100 text-green-600"
                : "bg-blue-50 text-blue-600 hover:bg-blue-100 active:bg-blue-200"
            }`}
          >
            {captured ? (
              <>
                <Check size={20} /> Captured
              </>
            ) : (
              <>
                <Crosshair size={20} /> Recapture
              </>
            )}
          </button>

          <button
            onClick={handleTest}
            disabled={testing}
            title="สั่งให้หุ่นวิ่งมายังตำแหน่งนี้"
            className={`flex-1 flex items-center justify-center gap-2.5 py-4 rounded-2xl font-bold text-base transition-all ${
              testing
                ? "bg-orange-100 text-orange-500 cursor-not-allowed"
                : "bg-orange-50 text-orange-600 hover:bg-orange-100 active:bg-orange-200"
            }`}
          >
            {testing ? (
              <>
                <Loader2 size={20} className="animate-spin" /> Moving...
              </>
            ) : (
              <>
                <Play size={20} fill="currentColor" /> Test
              </>
            )}
          </button>
          <button
            onClick={() => onShowTaskJson(idx)}
            title="ดู JSON ที่ส่งไป /goto_position"
            className="w-11 h-11 flex items-center justify-center text-gray-300 hover:text-purple-500 hover:bg-purple-50 rounded-2xl transition-colors flex-shrink-0"
          >
            <FileJson size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}

// --- Main Component ---
export default function JobEditor({
  mode,
  job,
  onSave,
  onCancel,
}: JobEditorProps) {
  const { jointStates, railPos, gripperPos, effectorPose, setTeachMode, sendGotoPosition } =
    useRos();
  const [jobName, setJobName] = useState(job?.name || "");
  const [jobDescription, setJobDescription] = useState(job?.description || "");
  const [tasks, setTasks] = useState<Task[]>(
    job?.tasks ? [...job.tasks] : []
  );
  const [isSaving, setIsSaving] = useState(false);
  const [activeTask, setActiveTask] = useState<Task | null>(null);
  const [showCaptureModal, setShowCaptureModal] = useState(false);
  const [captureLabel, setCaptureLabel] = useState("");
  const [toast, setToast] = useState<{ msg: string; ok: boolean } | null>(null);
  const [dryRunIdx, setDryRunIdx] = useState(-1);
  const [dryRunDelayMs, setDryRunDelayMs] = useState<number | null>(null);
  const dryRunDelayTotalRef = useRef(0);
  const [taskJsonIdx, setTaskJsonIdx] = useState<number | null>(null);
  const [taskJsonCopied, setTaskJsonCopied] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const dryRunCancelRef = useRef(false);

  const showToast = (msg: string, ok: boolean) => {
    setToast({ msg, ok });
    setTimeout(() => setToast(null), 3000);
  };

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
    useSensor(TouchSensor, {
      activationConstraint: { delay: 200, tolerance: 6 },
    })
  );

  useEffect(() => {
    setTeachMode(true);
    return () => {
      setTeachMode(false);
    };
  }, [setTeachMode]);

  const openCaptureModal = () => {
    setCaptureLabel(`Task ${tasks.length + 1}`);
    setShowCaptureModal(true);
  };

  const confirmCapture = () => {
    const newTask: Task = {
      id: Date.now(),
      sequence: tasks.length + 1,
      label: captureLabel || `Task ${tasks.length + 1}`,
      j1: jointStates[0],
      j2: jointStates[1],
      j3: jointStates[2],
      j4: jointStates[3],
      j5: jointStates[4],
      j6: jointStates[5],
      rail: railPos,
      gripper: gripperPos,
      speed: 50,
      delay: 2000,
      controlMode: "effector",
      x: effectorPose.x,
      y: effectorPose.y,
      z: effectorPose.z,
      roll: effectorPose.roll,
      pitch: effectorPose.pitch,
      yaw: effectorPose.yaw,
    };
    setTasks((prev) => [...prev, newTask]);
    setShowCaptureModal(false);
    setCaptureLabel("");
    // Scroll to newly added task
    setTimeout(() => {
      if (scrollContainerRef.current) {
        scrollContainerRef.current.scrollTo({
          top: scrollContainerRef.current.scrollHeight,
          behavior: "smooth",
        });
      }
    }, 150);
  };

  const deleteTask = (index: number) => {
    setTasks(tasks.filter((_, i) => i !== index));
  };

  const updateTaskLabel = (index: number, value: string) => {
    setTasks(tasks.map((t, i) => (i === index ? { ...t, label: value } : t)));
  };

  const updateTaskSpeed = (index: number, value: number) => {
    setTasks(tasks.map((t, i) => (i === index ? { ...t, speed: value } : t)));
  };

  const updateTaskDelay = (index: number, value: number) => {
    setTasks(tasks.map((t, i) => (i === index ? { ...t, delay: value } : t)));
  };

  const updateTaskControlMode = (id: number, mode: string) => {
    setTasks(tasks.map((t) => (t.id === id ? { ...t, controlMode: mode } : t)));
  };

  const recaptureTask = (index: number) => {
    setTasks(
      tasks.map((t, i) =>
        i === index
          ? {
              ...t,
              j1: jointStates[0],
              j2: jointStates[1],
              j3: jointStates[2],
              j4: jointStates[3],
              j5: jointStates[4],
              j6: jointStates[5],
              rail: railPos,
              gripper: gripperPos,
              x: effectorPose.x,
              y: effectorPose.y,
              z: effectorPose.z,
              roll: effectorPose.roll,
              pitch: effectorPose.pitch,
              yaw: effectorPose.yaw,
            }
          : t
      )
    );
  };

  const testTask = (index: number) => {
    const task = tasks[index];
    setTeachMode(false);
    sendGotoPosition({
      sequence: task.sequence,
      label: task.label,
      j1: task.j1,
      j2: task.j2,
      j3: task.j3,
      j4: task.j4,
      j5: task.j5,
      j6: task.j6,
      rail: task.rail,
      speed: task.speed ?? 50,
      gripper: task.gripper ?? 0,
    });
    const moveDuration = Math.max(2000, ((100 - (task.speed ?? 50)) / 100) * 5000);
    setTimeout(() => setTeachMode(true), moveDuration);
  };

  const startDryRun = async () => {
    if (tasks.length === 0) { showToast("ไม่มี Task ให้ทดสอบ", false); return; }
    dryRunCancelRef.current = false;
    setTeachMode(false);
    for (let i = 0; i < tasks.length; i++) {
      if (dryRunCancelRef.current) break;
      setDryRunIdx(i);
      const task = tasks[i];
      sendGotoPosition({
        sequence: i + 1,
        label: task.label,
        j1: task.j1, j2: task.j2, j3: task.j3,
        j4: task.j4, j5: task.j5, j6: task.j6,
        rail: task.rail,
        speed: task.speed ?? 50,
        gripper: task.gripper ?? 0,
      });
      // Movement wait — same formula as real execution
      const moveMs = Math.max(2000, ((100 - (task.speed ?? 50)) / 100) * 5000);
      await new Promise((r) => setTimeout(r, moveMs));

      // Delay countdown — visual, same as real execution
      const delay = task.delay ?? 0;
      if (delay > 0 && !dryRunCancelRef.current) {
        dryRunDelayTotalRef.current = delay;
        let remaining = delay;
        setDryRunDelayMs(remaining);
        while (remaining > 0 && !dryRunCancelRef.current) {
          await new Promise(r => setTimeout(r, 100));
          remaining -= 100;
          setDryRunDelayMs(Math.max(0, remaining));
        }
        setDryRunDelayMs(null);
      }
    }
    setTeachMode(true);
    setDryRunIdx(-1);
    if (!dryRunCancelRef.current) showToast(`Dry Run เสร็จ! (${tasks.length} tasks)`, true);
  };

  const handleStopDryRun = () => {
    dryRunCancelRef.current = true;
    setDryRunIdx(-1);
    setDryRunDelayMs(null);
    setTeachMode(true);
  };

  const handleDragStart = (event: DragStartEvent) => {
    const dragged = tasks.find((t) => t.id === event.active.id);
    setActiveTask(dragged || null);
  };

  const handleDragEnd = (event: DragEndEvent) => {
    setActiveTask(null);
    const { active, over } = event;
    if (!over || active.id === over.id) return;

    const oldIndex = tasks.findIndex((t) => t.id === active.id);
    const newIndex = tasks.findIndex((t) => t.id === over.id);
    const reordered = arrayMove(tasks, oldIndex, newIndex).map((t, i) => ({
      ...t,
      sequence: i + 1,
    }));
    setTasks(reordered);
  };

  const handleSave = async () => {
    if (!jobName.trim()) {
      showToast("กรุณาใส่ชื่องาน", false);
      return;
    }
    if (mode === "create" && tasks.length === 0) {
      showToast("กรุณา Capture ตำแหน่งอย่างน้อย 1 จุด", false);
      return;
    }

    setIsSaving(true);
    try {
      if (mode === "create") {
        const jobRes = await fetch("/api/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: jobName, description: jobDescription }),
        });
        if (!jobRes.ok) throw new Error("Failed to create job");
        const jobData = await jobRes.json();
        const newJobId = jobData.job.id;

        for (let i = 0; i < tasks.length; i++) {
          const taskRes = await fetch(`/api/jobs/${newJobId}/tasks`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ...tasks[i], sequence: i + 1,
              x: tasks[i].x ?? null, y: tasks[i].y ?? null, z: tasks[i].z ?? null,
              roll: tasks[i].roll ?? null, pitch: tasks[i].pitch ?? null, yaw: tasks[i].yaw ?? null,
            }),
          });
          if (!taskRes.ok) throw new Error("Failed to create task");
        }
        showToast("สร้างงานสำเร็จ!", true);
      } else {
        const updateRes = await fetch(`/api/jobs/${job?.id}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: jobName, description: jobDescription }),
        });
        if (!updateRes.ok) throw new Error("Failed to update job");

        const originalTaskIds = new Set((job?.tasks || []).map((t) => t.id));

        // Delete tasks removed in UI
        const deletedIds = (job?.tasks || [])
          .map((t) => t.id)
          .filter((id) => !tasks.some((t) => t.id === id));
        await Promise.all(
          deletedIds.map((id) =>
            fetch(`/api/jobs/${job?.id}/tasks/${id}`, { method: "DELETE" })
          )
        );

        // Save all tasks with correct sequence (position in current array)
        await Promise.all(
          tasks.map(async (t, i) => {
            const body = JSON.stringify({
              label: t.label,
              sequence: i + 1,
              j1: t.j1, j2: t.j2, j3: t.j3,
              j4: t.j4, j5: t.j5, j6: t.j6,
              rail: t.rail,
              speed: t.speed ?? 50,
              delay: t.delay ?? 0,
              gripper: t.gripper ?? 0,
              controlMode: t.controlMode ?? "joint",
              x: t.x ?? null, y: t.y ?? null, z: t.z ?? null,
              roll: t.roll ?? null, pitch: t.pitch ?? null, yaw: t.yaw ?? null,
            });
            const res = await (originalTaskIds.has(t.id)
              ? fetch(`/api/jobs/${job?.id}/tasks/${t.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body })
              : fetch(`/api/jobs/${job?.id}/tasks`, { method: "POST", headers: { "Content-Type": "application/json" }, body })
            );
            if (!res.ok) {
              const errData = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
              throw new Error(errData.error || `HTTP ${res.status}`);
            }
          })
        );
        showToast("บันทึกงานสำเร็จ!", true);
      }
      setTimeout(() => onSave(), 1200);
    } catch (error) {
      console.error("Error saving job:", error);
      showToast("บันทึกไม่สำเร็จ กรุณาลองอีกครั้ง", false);
    } finally {
      setIsSaving(false);
    }
  };

  const title = mode === "create" ? "New Job" : `Edit: ${jobName || job?.name}`;

  return (
    <div className="h-screen flex flex-col bg-[#F5F5F7]">
      {/* Top Bar */}
      <div className="p-6 bg-white border-b flex justify-between items-center sticky top-0 z-50">
        <button
          onClick={onCancel}
          className="flex items-center gap-2 text-gray-400 font-bold hover:text-black transition-colors"
        >
          <ArrowLeft size={24} /> Back
        </button>
        <RosStatusBadge />
        <div className="flex gap-4">
          <button
            onClick={() => sendGotoPosition({ sequence: 0, label: "Home", j1: 0, j2: 0, j3: 0, j4: 0, j5: 0, j6: 0, rail: 0, speed: 20, gripper: 0 })}
            disabled={dryRunIdx >= 0}
            className="p-3 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-xl transition-colors disabled:opacity-30"
            title="ส่งหุ่นยนต์กลับตำแหน่ง Home (ทุก Joint = 0)"
          >
            <Home size={22} />
          </button>
          <button
            onClick={dryRunIdx >= 0 ? handleStopDryRun : startDryRun}
            disabled={isSaving}
            className={`px-8 py-3 rounded-[24px] border font-bold disabled:opacity-50 flex items-center gap-2 transition-colors ${
              dryRunIdx >= 0
                ? "border-red-400 text-red-500 bg-red-50 hover:bg-red-100"
                : "border-gray-300 hover:border-gray-400"
            }`}
          >
            {dryRunIdx >= 0 ? (
              <><Square size={16} fill="currentColor" /> Stop ({dryRunIdx + 1}/{tasks.length})</>
            ) : (
              "Dry Run"
            )}
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="px-10 py-3 rounded-full bg-black text-white font-bold disabled:opacity-50 flex items-center gap-2"
          >
            <Save size={20} /> {isSaving ? "Saving..." : "Save Job"}
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel */}
        <div className="w-80 p-8 border-r bg-white space-y-8 overflow-y-auto">
          <h2 className="text-3xl font-black tracking-tight">{title}</h2>

          <textarea
            value={jobName}
            onChange={(e) => setJobName(e.target.value)}
            placeholder="ชื่องาน / Job Name..."
            className="w-full p-5 bg-gray-50 border-2 border-transparent rounded-[28px] text-xl font-bold placeholder-gray-300 focus:outline-none focus:border-blue-400 transition-colors resize-none h-24"
          />

          <textarea
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Description (optional)..."
            className="w-full p-5 bg-gray-50 border-2 border-transparent rounded-[28px] text-sm placeholder-gray-300 focus:outline-none focus:border-blue-400 transition-colors resize-none h-16"
          />

          <div className="p-6 bg-blue-50 text-blue-700 rounded-[30px] text-sm font-medium border border-blue-100">
            <div className="flex items-center gap-2 mb-2 font-bold">
              <Info size={18} /> Teaching Active
            </div>
            ลากปลายหุ่นไปยังจุดที่ต้องการ แล้วกดปุ่ม &quot;Capture Position&quot;
          </div>

          {mode === "edit" && (
            <div className="p-6 bg-gray-50 rounded-[30px] text-sm text-gray-500 border border-gray-100">
              <p className="font-bold mb-1">{tasks.length} Tasks</p>
              <p className="text-xs opacity-70">ลากแถบซ้ายของการ์ดเพื่อเรียงลำดับ</p>
            </div>
          )}

          <button
            onClick={openCaptureModal}
            className="w-full apple-btn bg-blue-600 text-white flex items-center justify-center gap-3 shadow-lg"
          >
            + Capture Position
          </button>
        </div>

        {/* Right Panel — Sortable Timeline */}
        <div
          ref={scrollContainerRef}
          className="flex-1 p-10 overflow-y-auto bg-gray-50/50"
        >
          <div className="mb-10">
            <h2 className="text-4xl font-black tracking-tight">Timeline</h2>
          </div>

          {tasks.length === 0 ? (
            <div className="text-center text-gray-400 py-10">
              No positions captured yet. Tap &apos;Capture Position&apos; to add tasks.
            </div>
          ) : (
            <DndContext
              sensors={sensors}
              collisionDetection={closestCenter}
              onDragStart={handleDragStart}
              onDragEnd={handleDragEnd}
            >
              <SortableContext
                items={tasks.map((t) => t.id)}
                strategy={verticalListSortingStrategy}
              >
                <div className="space-y-4">
                  {tasks.map((t, idx) => (
                    <SortableTaskCard
                      key={t.id}
                      task={t}
                      idx={idx}
                      onLabelChange={updateTaskLabel}
                      onSpeedChange={updateTaskSpeed}
                      onRecapture={recaptureTask}
                      onTest={testTask}
                      onDelete={deleteTask}
                      onDelayChange={updateTaskDelay}
                      onShowTaskJson={setTaskJsonIdx}
                      onControlModeChange={updateTaskControlMode}
                      isRunning={dryRunIdx === idx}
                      isDone={dryRunIdx > idx && dryRunIdx >= 0}
                    />
                  ))}
                </div>
              </SortableContext>

              <DragOverlay>
                {activeTask ? (
                  <div className="tesla-card p-5 flex items-center gap-4 shadow-2xl rotate-1 scale-[1.02] opacity-95">
                    <div className="p-2 text-gray-300">
                      <GripHorizontal size={22} />
                    </div>
                    <div className="w-10 h-10 bg-blue-100 rounded-[14px] flex items-center justify-center font-black text-blue-500 text-sm flex-shrink-0">
                      {tasks.findIndex((t) => t.id === activeTask.id) + 1}
                    </div>
                    <span className="flex-1 font-bold text-xl text-[#1D1D1F]">
                      {activeTask.label || `Task`}
                    </span>
                    <div className="text-xs font-mono font-bold text-gray-400 uppercase hidden lg:block">
                      J:{" "}
                      {[
                        activeTask.j1,
                        activeTask.j2,
                        activeTask.j3,
                        activeTask.j4,
                        activeTask.j5,
                        activeTask.j6,
                      ]
                        .map((v) => v.toFixed(0))
                        .join(", ")}{" "}
                      | R: {activeTask.rail.toFixed(0)}mm
                    </div>
                  </div>
                ) : null}
              </DragOverlay>
            </DndContext>
          )}
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div className={`fixed top-6 left-1/2 -translate-x-1/2 z-[200] flex items-center gap-3 px-8 py-4 rounded-full font-bold text-white shadow-2xl transition-all ${toast.ok ? "bg-green-500" : "bg-red-500"}`}>
          {toast.ok ? <CheckCircle2 size={20} /> : <AlertCircle size={20} />}
          {toast.msg}
        </div>
      )}

      {/* Dry Run — Delay Countdown */}
      {dryRunDelayMs !== null && dryRunDelayMs > 0 && (
        <div className="fixed inset-0 z-190 flex items-center justify-center pointer-events-none">
          <div className="bg-gray-900/95 backdrop-blur-md rounded-3xl px-16 py-12 shadow-2xl text-center min-w-75">
            <p className="text-white/50 text-xs font-black uppercase tracking-widest mb-4">
              Task {dryRunIdx + 1} เสร็จแล้ว — รอก่อน
            </p>
            <p className="text-orange-400 font-black tabular-nums leading-none" style={{ fontSize: "80px" }}>
              {(dryRunDelayMs / 1000).toFixed(1)}
            </p>
            <p className="text-white/40 text-sm mt-1">วินาที</p>
            <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden mt-6">
              <div
                className="h-full bg-orange-400 rounded-full transition-none"
                style={{ width: `${(dryRunDelayMs / dryRunDelayTotalRef.current) * 100}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Task JSON Preview Modal */}
      {taskJsonIdx !== null && tasks[taskJsonIdx] && (() => {
        const t = tasks[taskJsonIdx];
        const payload = {
          sequence: taskJsonIdx + 1,
          label: t.label ?? `Task ${taskJsonIdx + 1}`,
          j1: t.j1, j2: t.j2, j3: t.j3,
          j4: t.j4, j5: t.j5, j6: t.j6,
          rail: t.rail,
          speed: t.speed ?? 50,
          gripper: t.gripper ?? 0,
        };
        const jsonStr = JSON.stringify(payload, null, 2);
        const handleCopy = async () => {
          await navigator.clipboard.writeText(jsonStr);
          setTaskJsonCopied(true);
          setTimeout(() => setTaskJsonCopied(false), 2000);
        };
        const handleDownload = () => {
          const now = new Date();
          const ts = `${now.getFullYear()}${String(now.getMonth()+1).padStart(2,"0")}${String(now.getDate()).padStart(2,"0")}_${String(now.getHours()).padStart(2,"0")}${String(now.getMinutes()).padStart(2,"0")}${String(now.getSeconds()).padStart(2,"0")}`;
          const safeJob = (jobName||"job").replace(/[^a-zA-Z0-9ก-๙\s-]/g,"").trim().replace(/\s+/g,"_");
          const safeTask = (t.label||`Task${taskJsonIdx+1}`).replace(/[^a-zA-Z0-9ก-๙\s-]/g,"").trim().replace(/\s+/g,"_");
          const blob = new Blob([jsonStr], { type: "application/json" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url; a.download = `${safeJob}_${safeTask}_${ts}.json`; a.click();
          URL.revokeObjectURL(url);
        };
        return (
          <div
            className="fixed inset-0 z-[150] flex items-center justify-center p-6 bg-black/50 backdrop-blur-sm"
            onClick={() => setTaskJsonIdx(null)}
          >
            <div
              className="w-full max-w-md bg-[#1E1E2E] rounded-3xl overflow-hidden shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
                <div>
                  <p className="text-white font-black text-sm">/goto_position payload</p>
                  <p className="text-white/40 text-xs">Task {taskJsonIdx + 1} — {t.label || `Task ${taskJsonIdx + 1}`}</p>
                </div>
                <div className="flex items-center gap-2">
                  <button onClick={handleDownload} className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/10 hover:bg-white/20 text-white text-xs font-bold transition-colors">
                    <Download size={13} /> Download
                  </button>
                  <button onClick={handleCopy} className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/10 hover:bg-white/20 text-white text-xs font-bold transition-colors">
                    {taskJsonCopied ? <Check size={13} className="text-green-400" /> : <Copy size={13} />}
                    {taskJsonCopied ? "Copied!" : "Copy"}
                  </button>
                  <button onClick={() => setTaskJsonIdx(null)} className="p-1.5 rounded-xl bg-white/10 hover:bg-white/20 text-white transition-colors">
                    <X size={16} />
                  </button>
                </div>
              </div>
              <pre className="p-6 text-xs font-mono text-green-300 overflow-auto max-h-72">
                {jsonStr}
              </pre>
            </div>
          </div>
        );
      })()}

      {/* Capture Position Modal */}
      {showCaptureModal && (
        <div
          className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-6"
          onClick={() => setShowCaptureModal(false)}
        >
          <div
            className="bg-white rounded-[32px] p-8 w-full max-w-md shadow-2xl space-y-6"
            onClick={(e) => e.stopPropagation()}
          >
            <div>
              <h2 className="text-2xl font-black">Capture Position</h2>
              <p className="text-gray-400 text-sm mt-1">
                บันทึกตำแหน่งปัจจุบันของหุ่นยนต์
              </p>
            </div>

            {/* Current position preview */}
            <div className="space-y-2">
              <div className="bg-gray-50 rounded-2xl p-4 grid grid-cols-2 gap-2 text-xs font-mono text-gray-500">
                <div className="flex items-center gap-1.5">
                  <RotateCcw size={11} className="opacity-50 flex-shrink-0" />
                  <span>
                    J1-J3: {jointStates[0].toFixed(1)}° /{" "}
                    {jointStates[1].toFixed(1)}° / {jointStates[2].toFixed(1)}°
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <RotateCcw size={11} className="opacity-50 flex-shrink-0" />
                  <span>
                    J4-J6: {jointStates[3].toFixed(1)}° /{" "}
                    {jointStates[4].toFixed(1)}° / {jointStates[5].toFixed(1)}°
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <MoveHorizontal size={11} className="opacity-50 flex-shrink-0" />
                  <span>Rail: {railPos.toFixed(1)} mm</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Hand size={11} className="opacity-50 flex-shrink-0" />
                  <span>Grip: {gripperPos}%</span>
                </div>
              </div>
              {/* Effector pose preview */}
              <div className="bg-purple-50 rounded-2xl p-3">
                <p className="text-[10px] font-black text-purple-400 uppercase mb-2">End-Effector Pose</p>
                <div className="grid grid-cols-3 gap-1.5 text-xs font-mono text-purple-600">
                  <div><span className="opacity-50">X </span>{effectorPose.x.toFixed(1)}</div>
                  <div><span className="opacity-50">Y </span>{effectorPose.y.toFixed(1)}</div>
                  <div><span className="opacity-50">Z </span>{effectorPose.z.toFixed(1)}</div>
                  <div><span className="opacity-50">R </span>{effectorPose.roll.toFixed(1)}°</div>
                  <div><span className="opacity-50">P </span>{effectorPose.pitch.toFixed(1)}°</div>
                  <div><span className="opacity-50">Yw </span>{effectorPose.yaw.toFixed(1)}°</div>
                </div>
              </div>
            </div>

            {/* Task name input */}
            <input
              type="text"
              value={captureLabel}
              onChange={(e) => setCaptureLabel(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && confirmCapture()}
              placeholder="ชื่อ Task..."
              autoFocus
              className="w-full p-5 bg-gray-50 rounded-2xl text-xl font-bold focus:outline-none focus:ring-2 focus:ring-blue-400 border-2 border-transparent"
            />

            <div className="flex gap-3">
              <button
                onClick={() => setShowCaptureModal(false)}
                className="flex-1 py-4 rounded-2xl bg-gray-100 font-bold text-gray-600 text-lg hover:bg-gray-200 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmCapture}
                className="flex-1 py-4 rounded-2xl bg-blue-600 text-white font-black text-lg hover:bg-blue-700 transition-colors"
              >
                ✓ Save Task
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
