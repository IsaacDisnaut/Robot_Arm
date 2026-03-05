"use client";

import React, { useEffect, useState, useCallback } from "react";
import { useRos } from "@/context/RosContext";
import { Pause, Play, X } from "lucide-react";

interface Task {
  id: number;
  sequence: number;
  label?: string;
  speed: number;
  delay: number;
}

interface ExecutionMonitorProps {
  tasks: Task[];
  totalEstimatedTime?: number; // in seconds
  onComplete?: () => void;
  onStop?: () => void;
}

export default function ExecutionMonitor({
  tasks,
  totalEstimatedTime = 60,
  onComplete,
  onStop,
}: ExecutionMonitorProps) {
  const { isExecuting, executionStartTime, currentTaskIndex, stopExecution } =
    useRos();
  const [elapsedTime, setElapsedTime] = useState(0);
  const [taskProgress, setTaskProgress] = useState<number[]>(
    tasks.map(() => 0)
  );

  // Update elapsed time
  useEffect(() => {
    if (!isExecuting || !executionStartTime) return;

    const interval = setInterval(() => {
      const elapsed = (Date.now() - executionStartTime) / 1000;
      setElapsedTime(elapsed);

      // Calculate progress for each task
      // Distribute time equally across tasks
      const timePerTask = totalEstimatedTime / tasks.length;
      const newProgress = tasks.map((_, index) => {
        const taskStartTime = index * timePerTask;
        const taskEndTime = (index + 1) * timePerTask;

        if (elapsed < taskStartTime) return 0;
        if (elapsed >= taskEndTime) return 100;

        const taskElapsed = elapsed - taskStartTime;
        return Math.round((taskElapsed / timePerTask) * 100);
      });

      setTaskProgress(newProgress);

      // Check if execution completed
      if (elapsed >= totalEstimatedTime) {
        clearInterval(interval);
        onComplete?.();
      }
    }, 200);

    return () => clearInterval(interval);
  }, [isExecuting, executionStartTime, tasks.length, totalEstimatedTime, onComplete]);

  const overallProgress = Math.round(
    (elapsedTime / totalEstimatedTime) * 100
  );
  const minutes = Math.floor(elapsedTime / 60);
  const seconds = Math.floor(elapsedTime % 60);
  const remainingSeconds = Math.max(0, totalEstimatedTime - elapsedTime);
  const remainingMinutes = Math.floor(remainingSeconds / 60);
  const remainingSecs = Math.floor(remainingSeconds % 60);

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-lg z-[300] flex items-center justify-center p-6">
      <div className="bg-white rounded-[48px] max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 p-8 flex justify-between items-center rounded-t-[48px]">
          <div>
            <h2 className="text-4xl font-black mb-2">executing job</h2>
            <p className="text-gray-400 text-lg">
              {minutes}:{seconds.toString().padStart(2, "0")} elapsed
            </p>
          </div>
          <button
            onClick={() => {
              stopExecution();
              onStop?.();
            }}
            className="p-3 hover:bg-gray-100 rounded-full transition-colors"
          >
            <X size={32} className="text-gray-600" />
          </button>
        </div>

        {/* Overall Progress */}
        <div className="p-8 border-b border-gray-200">
          <div className="flex justify-between items-end mb-3">
            <h3 className="text-sm font-bold text-gray-400 uppercase">
              overall progress
            </h3>
            <span className="text-4xl font-black text-blue-600">
              {overallProgress}%
            </span>
          </div>
          <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-300"
              style={{ width: `${Math.min(100, overallProgress)}%` }}
            />
          </div>
          <p className="text-xs text-gray-400 mt-3">
            time remaining: {remainingMinutes}:{remainingSecs.toString().padStart(2, "0")}
          </p>
        </div>

        {/* Task List */}
        <div className="p-8 space-y-4">
          <h3 className="text-xl font-bold text-gray-900 mb-6">tasks</h3>
          {tasks.map((task, index) => {
            const isCompleted = taskProgress[index] === 100;
            const isActive = index === currentTaskIndex && isExecuting;
            const isPending = index > currentTaskIndex;

            return (
              <div
                key={task.id}
                className={`p-6 rounded-[32px] border-2 transition-all ${
                  isActive
                    ? "border-blue-500 bg-blue-50"
                    : isCompleted
                      ? "border-green-500 bg-green-50"
                      : isPending
                        ? "border-gray-200 bg-gray-50"
                        : "border-gray-200 bg-gray-50"
                }`}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-4">
                    <div
                      className={`w-12 h-12 rounded-[16px] flex items-center justify-center font-bold text-lg ${
                        isCompleted
                          ? "bg-green-500 text-white"
                          : isActive
                            ? "bg-blue-500 text-white"
                            : "bg-gray-300 text-white"
                      }`}
                    >
                      {isCompleted ? "✓" : isActive ? "▶" : index + 1}
                    </div>
                    <div>
                      <p className="font-bold text-lg text-gray-900">
                        {task.label || `Task ${task.sequence}`}
                      </p>
                      <p className="text-sm text-gray-500">
                        Speed: {task.speed}% | Delay: {task.delay}ms
                      </p>
                    </div>
                  </div>
                  <span
                    className={`text-2xl font-black ${
                      isActive
                        ? "text-blue-600"
                        : isCompleted
                          ? "text-green-600"
                          : "text-gray-400"
                    }`}
                  >
                    {taskProgress[index]}%
                  </span>
                </div>

                {/* Progress Bar */}
                <div className="w-full h-2 bg-gray-300 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      isCompleted
                        ? "bg-green-500"
                        : isActive
                          ? "bg-blue-500"
                          : "bg-gray-400"
                    }`}
                    style={{ width: `${taskProgress[index]}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 p-8 border-t border-gray-200 bg-white rounded-b-[48px] flex gap-4">
          <button
            onClick={() => {
              stopExecution();
              onStop?.();
            }}
            className="flex-1 px-6 py-4 bg-red-500 hover:bg-red-600 text-white font-bold rounded-[24px] transition-colors flex items-center justify-center gap-2"
          >
            <X size={20} /> Stop
          </button>
        </div>
      </div>
    </div>
  );
}
