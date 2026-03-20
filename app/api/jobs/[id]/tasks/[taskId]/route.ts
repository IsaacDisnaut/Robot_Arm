import { prisma } from "@/lib/prisma";
import { NextResponse } from "next/server";

export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ taskId: string }> }
) {
  try {
    const { taskId } = await params;
    const { label, sequence, j1, j2, j3, j4, j5, j6, rail, speed, delay, gripper, controlMode, x, y, z, roll, pitch, yaw } = await req.json();

    const task = await prisma.task.update({
      where: { id: parseInt(taskId) },
      data: {
        ...(label !== undefined && { label: label ?? null }),
        ...(sequence !== undefined && { sequence }),
        ...(j1 !== undefined && { j1 }),
        ...(j2 !== undefined && { j2 }),
        ...(j3 !== undefined && { j3 }),
        ...(j4 !== undefined && { j4 }),
        ...(j5 !== undefined && { j5 }),
        ...(j6 !== undefined && { j6 }),
        ...(rail !== undefined && { rail }),
        ...(speed !== undefined && { speed }),
        ...(delay !== undefined && { delay }),
        ...(gripper !== undefined && { gripper }),
        ...(controlMode !== undefined && { controlMode }),
        ...(x !== undefined && { x: x ?? null }),
        ...(y !== undefined && { y: y ?? null }),
        ...(z !== undefined && { z: z ?? null }),
        ...(roll !== undefined && { roll: roll ?? null }),
        ...(pitch !== undefined && { pitch: pitch ?? null }),
        ...(yaw !== undefined && { yaw: yaw ?? null }),
      },
    });

    return NextResponse.json({ success: true, task });
  } catch (error) {
    console.error("Error updating task:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ taskId: string }> }
) {
  try {
    const { taskId } = await params;
    const parsedTaskId = parseInt(taskId);
    await prisma.task.delete({
      where: { id: parsedTaskId },
    });

    return NextResponse.json({
      success: true,
      message: "Task deleted successfully",
    });
  } catch (error) {
    console.error("Error deleting task:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
