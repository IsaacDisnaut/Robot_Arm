"use client";
import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useCallback,
} from "react";
import * as ROSLIB from "roslib";

export interface CalibrationData {
  offsets: number[];  // [j1, j2, j3, j4, j5, j6, rail, gripper]
  flips: boolean[];   // [j1, j2, j3, j4, j5, j6, rail, gripper]
  tcpOffset: { x: number; y: number; z: number };  // Tool center point offset in mm
}

const DEFAULT_CALIBRATION: CalibrationData = {
  offsets: [0, 0, 0, 0, 0, 0, 0, 0],
  flips: [false, false, false, false, false, false, false, false],
  tcpOffset: { x: 0, y: 0, z: 0 },
};

export interface EffectorPose {
  x: number; y: number; z: number;
  roll: number; pitch: number; yaw: number;
}

interface RosContextType {
  isConnected: boolean;
  jointStates: number[];
  jointVelocities: number[];
  railPos: number;
  gripperPos: number;
  safetyStatus: number;
  robotStatus: number;
  effectorPose: EffectorPose;
  sendJob: (jobData: any) => void;
  sendGotoPosition: (taskData: any) => void;
  setTeachMode: (status: boolean) => void;
  isExecuting: boolean;
  isPaused: boolean;
  executionStartTime: number | null;
  currentTaskIndex: number;
  stopExecution: () => void;
  pauseExecution: () => void;
  resumeExecution: () => void;
  startExecution: () => void;
  setCurrentTaskIndex: (index: number) => void;
  isTestMode: boolean;
  setTestMode: (v: boolean) => void;
  calibration: CalibrationData;
  setCalibration: (data: CalibrationData) => void;
}

const RosContext = createContext<RosContextType | null>(null);

export const RosProvider = ({ children }: { children: React.ReactNode }) => {
  const [ros, setRos] = useState<ROSLIB.Ros | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [jointStates, setJointStates] = useState([0, 0, 0, 0, 0, 0]);
  const [jointVelocities, setJointVelocities] = useState([0, 0, 0, 0, 0, 0]);
  const [railPos, setRailPos] = useState(0);
  const [gripperPos, setGripperPos] = useState(0);
  const [safetyStatus, setSafetyStatus] = useState(0);
  const [robotStatus, setRobotStatus] = useState(0);
  const [effectorPose, setEffectorPose] = useState<EffectorPose>({ x: 0, y: 0, z: 0, roll: 0, pitch: 0, yaw: 0 });
  const [isExecuting, setIsExecuting] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [executionStartTime, setExecutionStartTime] = useState<number | null>(
    null
  );
  const [currentTaskIndex, setCurrentTaskIndex] = useState(0);
  const [isTestMode, setIsTestMode] = useState(false);

  const [calibration, setCalibrationState] = useState<CalibrationData>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("robotCalibration");
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          // Migrate old calibration data that doesn't have tcpOffset
          if (!parsed.tcpOffset) parsed.tcpOffset = { x: 0, y: 0, z: 0 };
          return parsed;
        } catch {}
      }
    }
    return DEFAULT_CALIBRATION;
  });

  // Ref so subscriber closure always sees latest calibration without re-subscribing
  const calibrationRef = useRef(calibration);
  useEffect(() => { calibrationRef.current = calibration; }, [calibration]);

  const setCalibration = useCallback((data: CalibrationData) => {
    setCalibrationState(data);
    localStorage.setItem("robotCalibration", JSON.stringify(data));
  }, []);

  // Forward transform: raw sensor → display value
  const applyForward = (raw: number, index: number, cal: CalibrationData) =>
    (cal.flips[index] ? -raw : raw) + cal.offsets[index];

  // Inverse transform: display value → raw command
  const applyInverse = useCallback((display: number, index: number) => {
    const cal = calibrationRef.current;
    const unOffset = display - cal.offsets[index];
    return cal.flips[index] ? -unOffset : unOffset;
  }, []);

  useEffect(() => {
    const connectRos = () => {
      const rosInstance = new ROSLIB.Ros({
        url: process.env.NEXT_PUBLIC_ROS_URL ?? "ws://localhost:9090",
      });

      rosInstance.on("connection", () => {
        console.log("ROS Bridge Connected");
        setIsConnected(true);
      });

      rosInstance.on("error", () => setIsConnected(false));

      rosInstance.on("close", () => {
        setIsConnected(false);
        setTimeout(connectRos, 5000);
      });

      // Joint state subscriber
      const jointSub = new ROSLIB.Topic({
        ros: rosInstance,
        name: "/joint_states",
        messageType: "sensor_msgs/JointState",
      });
      jointSub.subscribe((m: any) => {
        const cal = calibrationRef.current;
        if (m.position) {
          const calibratedJoints = m.position
            .slice(0, 6)
            .map((raw: number, i: number) => applyForward(raw, i, cal));
          setJointStates(calibratedJoints);
          if (m.position[6] !== undefined)
            setRailPos(applyForward(m.position[6], 6, cal));
          if (m.position[7] !== undefined)
            setGripperPos(applyForward(m.position[7], 7, cal));
        }
        if (m.velocity && m.velocity.length >= 6) {
          setJointVelocities(m.velocity.slice(0, 6));
        }
      });

      // End-effector pose subscriber (XYZ mm + RPY degrees)
      const poseSub = new ROSLIB.Topic({
        ros: rosInstance,
        name: "/end_effector_pose",
        messageType: "std_msgs/String",
      });
      poseSub.subscribe((m: any) => {
        try {
          const p = JSON.parse(m.data);
          setEffectorPose({
            x: p.x ?? 0, y: p.y ?? 0, z: p.z ?? 0,
            roll: p.roll ?? 0, pitch: p.pitch ?? 0, yaw: p.yaw ?? 0,
          });
        } catch {}
      });

      // Safety status subscriber
      const safetySub = new ROSLIB.Topic({
        ros: rosInstance,
        name: "/safety_status",
        messageType: "std_msgs/Int8",
      });
      safetySub.subscribe((m: any) => setSafetyStatus(m.data));

      // Robot execution status subscriber
      const robotStatusSub = new ROSLIB.Topic({
        ros: rosInstance,
        name: "/robot_status",
        messageType: "std_msgs/Int8",
      });
      robotStatusSub.subscribe((m: any) => {
        setRobotStatus(m.data);
        if (m.data === 0) {
          setIsExecuting(false);
          setIsPaused(false);
        } else if (m.data === 1) {
          setIsPaused(false);
        } else if (m.data === 2) {
          setIsPaused(true);
        }
      });

      setRos(rosInstance);
    };

    connectRos();
    return () => ros?.close();
  }, []);

  // Publish /tool_config whenever TCP offset changes
  useEffect(() => {
    if (!ros) return;
    const topic = new ROSLIB.Topic({
      ros,
      name: "/tool_config",
      messageType: "std_msgs/String",
    });
    topic.publish({
      data: JSON.stringify({
        tcp_x: calibration.tcpOffset.x,
        tcp_y: calibration.tcpOffset.y,
        tcp_z: calibration.tcpOffset.z,
      }),
    });
  }, [calibration.tcpOffset, ros]);

  const sendJob = useCallback(
    (jobData: any) => {
      if (!ros) return;
      const rawTasks = jobData.tasks?.map((task: any) => ({
        ...task,
        j1: applyInverse(task.j1, 0),
        j2: applyInverse(task.j2, 1),
        j3: applyInverse(task.j3, 2),
        j4: applyInverse(task.j4, 3),
        j5: applyInverse(task.j5, 4),
        j6: applyInverse(task.j6, 5),
        rail: applyInverse(task.rail, 6),
        gripper: applyInverse(task.gripper, 7),
      }));
      const topic = new ROSLIB.Topic({
        ros,
        name: "/execute_trajectory",
        messageType: "std_msgs/String",
      });
      topic.publish({ data: JSON.stringify({ ...jobData, tasks: rawTasks }) });
    },
    [ros, applyInverse]
  );

  const sendGotoPosition = useCallback(
    (taskData: any) => {
      if (!ros) return;
      const rawTask = {
        ...taskData,
        j1: applyInverse(taskData.j1, 0),
        j2: applyInverse(taskData.j2, 1),
        j3: applyInverse(taskData.j3, 2),
        j4: applyInverse(taskData.j4, 3),
        j5: applyInverse(taskData.j5, 4),
        j6: applyInverse(taskData.j6, 5),
        rail: applyInverse(taskData.rail, 6),
        gripper: applyInverse(taskData.gripper, 7),
        // Effector mode: include Cartesian target for robot IK
        ...(taskData.controlMode === "effector" && taskData.x != null && {
          x: taskData.x, y: taskData.y, z: taskData.z,
          roll: taskData.roll, pitch: taskData.pitch, yaw: taskData.yaw,
        }),
      };
      const topic = new ROSLIB.Topic({
        ros,
        name: "/goto_position",
        messageType: "std_msgs/String",
      });
      topic.publish({ data: JSON.stringify(rawTask) });
    },
    [ros, applyInverse]
  );

  const setTeachMode = useCallback(
    (status: boolean) => {
      if (!ros) return;
      const topic = new ROSLIB.Topic({
        ros,
        name: "/teach_mode",
        messageType: "std_msgs/Bool",
      });
      topic.publish({ data: status });
    },
    [ros]
  );

  const stopExecution = useCallback(() => {
    if (ros) {
      const topic = new ROSLIB.Topic({
        ros,
        name: "/stop_execution",
        messageType: "std_msgs/Bool",
      });
      topic.publish({ data: true });
    }
    setIsExecuting(false);
    setIsPaused(false);
    setExecutionStartTime(null);
    setCurrentTaskIndex(0);
  }, [ros]);

  const pauseExecution = useCallback(() => {
    if (ros) {
      const topic = new ROSLIB.Topic({
        ros,
        name: "/pause_execution",
        messageType: "std_msgs/Bool",
      });
      topic.publish({ data: true });
    }
    setIsPaused(true);
  }, [ros]);

  const resumeExecution = useCallback(() => {
    if (ros) {
      const topic = new ROSLIB.Topic({
        ros,
        name: "/pause_execution",
        messageType: "std_msgs/Bool",
      });
      topic.publish({ data: false });
    }
    setIsPaused(false);
  }, [ros]);

  const startExecution = useCallback(() => {
    setIsExecuting(true);
    setExecutionStartTime(Date.now());
    setCurrentTaskIndex(0);
  }, []);

  const updateCurrentTaskIndex = useCallback((index: number) => {
    setCurrentTaskIndex(index);
  }, []);

  return (
    <RosContext.Provider
      value={{
        isConnected,
        jointStates,
        jointVelocities,
        railPos,
        gripperPos,
        safetyStatus,
        robotStatus,
        effectorPose,
        sendJob,
        sendGotoPosition,
        setTeachMode,
        isExecuting,
        isPaused,
        executionStartTime,
        currentTaskIndex,
        stopExecution,
        pauseExecution,
        resumeExecution,
        startExecution,
        setCurrentTaskIndex: updateCurrentTaskIndex,
        isTestMode,
        setTestMode: setIsTestMode,
        calibration,
        setCalibration,
      }}
    >
      {children}
    </RosContext.Provider>
  );
};

export const useRos = () => {
  const context = useContext(RosContext);
  if (!context) throw new Error("useRos must be used within RosProvider");
  return context;
};
