import { prisma } from "@/lib/prisma";
import { NextResponse } from "next/server";

export async function GET() {
  try {
    // ลองอ่านจำนวน jobs
    const jobs = await prisma.job.findMany({
      include: { tasks: { orderBy: { sequence: "asc" } } },
    });

    return NextResponse.json({
      success: true,
      message: "Database connected successfully",
      jobsCount: jobs.length,
      jobs: jobs,
    });
  } catch (error) {
    console.error("Database error:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Database connection failed",
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

export async function POST(req: Request) {
  try {
    const { name, description } = await req.json();

    const job = await prisma.job.create({
      data: {
        name,
        description,
      },
    });

    return NextResponse.json(
      {
        success: true,
        message: "Job created successfully",
        job,
      },
      { status: 201 }
    );
  } catch (error) {
    console.error("Error creating job:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Failed to create job",
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
