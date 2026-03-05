import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

// Realistic robot arm sequences for a café environment
// Joints in degrees: J1=base rotation, J2=shoulder, J3=elbow, J4=wrist1, J5=wrist2, J6=wrist3
// Rail in mm (0-500mm linear rail)
// Speed: 10-100%, Delay: ms to wait after reaching position

const jobSequences = [
  {
    name: "Coffee Preparation",
    description: "ลำดับการชงกาแฟ: หยิบแก้ว → เครื่องชงกาแฟ → วางบนถาด",
    tasks: [
      { label: "Home Position", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
      { label: "Move to Cup Dispenser", j1: 42, j2: -55, j3: 25, j4: -35, j5: 5, j6: 15, rail: 60, speed: 70, delay: 100, gripper: 0 },
      { label: "Lower to Cup", j1: 42, j2: -82, j3: 48, j4: -28, j5: 5, j6: 15, rail: 60, speed: 30, delay: 200, gripper: 0 },
      { label: "Grip Cup", j1: 42, j2: -82, j3: 48, j4: -28, j5: 5, j6: 15, rail: 60, speed: 20, delay: 500, gripper: 80 },
      { label: "Lift Cup Up", j1: 42, j2: -55, j3: 25, j4: -35, j5: 5, j6: 15, rail: 60, speed: 40, delay: 300, gripper: 80 },
      { label: "Move to Coffee Machine", j1: 88, j2: -62, j3: 30, j4: -58, j5: 0, j6: 30, rail: 220, speed: 65, delay: 100, gripper: 80 },
      { label: "Position Cup at Spout", j1: 88, j2: -88, j3: 52, j4: -42, j5: 0, j6: 30, rail: 220, speed: 20, delay: 300, gripper: 80 },
      { label: "Wait for Coffee", j1: 88, j2: -88, j3: 52, j4: -42, j5: 0, j6: 30, rail: 220, speed: 10, delay: 4000, gripper: 80 },
      { label: "Lift Full Cup", j1: 88, j2: -62, j3: 30, j4: -58, j5: 0, j6: 30, rail: 220, speed: 25, delay: 300, gripper: 80 },
      { label: "Move to Serving Tray", j1: 128, j2: -68, j3: 35, j4: -48, j5: -5, j6: 45, rail: 350, speed: 55, delay: 100, gripper: 80 },
      { label: "Lower Cup to Tray", j1: 128, j2: -90, j3: 55, j4: -40, j5: -5, j6: 45, rail: 350, speed: 20, delay: 300, gripper: 80 },
      { label: "Release Cup", j1: 128, j2: -90, j3: 55, j4: -40, j5: -5, j6: 45, rail: 350, speed: 10, delay: 400, gripper: 0 },
      { label: "Return Home", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
    ],
  },
  {
    name: "Pastry Placement",
    description: "วางขนมบนจาน: หยิบจากถาดเก็บ → จัดเรียงบนจานเสิร์ฟ",
    tasks: [
      { label: "Home Position", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
      { label: "Move Over Pastry Tray", j1: -35, j2: -50, j3: 20, j4: -40, j5: 8, j6: -20, rail: 80, speed: 65, delay: 100, gripper: 0 },
      { label: "Descend to Pastry", j1: -35, j2: -78, j3: 45, j4: -35, j5: 8, j6: -20, rail: 80, speed: 25, delay: 200, gripper: 0 },
      { label: "Grip Pastry", j1: -35, j2: -78, j3: 45, j4: -35, j5: 8, j6: -20, rail: 80, speed: 15, delay: 600, gripper: 80 },
      { label: "Lift Pastry", j1: -35, j2: -50, j3: 20, j4: -40, j5: 8, j6: -20, rail: 80, speed: 30, delay: 300, gripper: 80 },
      { label: "Rotate to Serving Plate", j1: 55, j2: -52, j3: 22, j4: -42, j5: -5, j6: 0, rail: 180, speed: 60, delay: 100, gripper: 80 },
      { label: "Align Over Plate", j1: 55, j2: -65, j3: 35, j4: -38, j5: -5, j6: 0, rail: 180, speed: 25, delay: 200, gripper: 80 },
      { label: "Lower Gently to Plate", j1: 55, j2: -85, j3: 50, j4: -32, j5: -5, j6: 0, rail: 180, speed: 15, delay: 300, gripper: 80 },
      { label: "Release Pastry", j1: 55, j2: -85, j3: 50, j4: -32, j5: -5, j6: 0, rail: 180, speed: 10, delay: 400, gripper: 0 },
      { label: "Lift Away from Plate", j1: 55, j2: -65, j3: 35, j4: -38, j5: -5, j6: 0, rail: 180, speed: 40, delay: 200, gripper: 0 },
      { label: "Return Home", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
    ],
  },
  {
    name: "Cup Stacking",
    description: "จัดเรียงแก้ว: หยิบแก้วซ้อนทับให้เป็นระเบียบ",
    tasks: [
      { label: "Home Position", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
      { label: "Approach Cup Stack A", j1: -60, j2: -52, j3: 22, j4: -38, j5: 0, j6: -10, rail: 120, speed: 70, delay: 100, gripper: 0 },
      { label: "Descend to Top Cup", j1: -60, j2: -80, j3: 50, j4: -32, j5: 0, j6: -10, rail: 120, speed: 25, delay: 200, gripper: 0 },
      { label: "Grip Top Cup", j1: -60, j2: -80, j3: 50, j4: -32, j5: 0, j6: -10, rail: 120, speed: 15, delay: 500, gripper: 80 },
      { label: "Lift Cup", j1: -60, j2: -52, j3: 22, j4: -38, j5: 0, j6: -10, rail: 120, speed: 35, delay: 200, gripper: 80 },
      { label: "Move to Target Stack B", j1: 60, j2: -52, j3: 22, j4: -38, j5: 0, j6: 10, rail: 280, speed: 65, delay: 100, gripper: 80 },
      { label: "Align Over Stack B", j1: 60, j2: -65, j3: 35, j4: -36, j5: 0, j6: 10, rail: 280, speed: 20, delay: 200, gripper: 80 },
      { label: "Lower Cup onto Stack", j1: 60, j2: -82, j3: 52, j4: -30, j5: 0, j6: 10, rail: 280, speed: 15, delay: 300, gripper: 80 },
      { label: "Release Cup", j1: 60, j2: -82, j3: 52, j4: -30, j5: 0, j6: 10, rail: 280, speed: 10, delay: 400, gripper: 0 },
      { label: "Lift Clear of Stack", j1: 60, j2: -55, j3: 25, j4: -38, j5: 0, j6: 10, rail: 280, speed: 45, delay: 200, gripper: 0 },
      { label: "Return Home", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
    ],
  },
  {
    name: "Tray Organization",
    description: "จัดถาดเสิร์ฟ: เรียงแก้ว จาน และอุปกรณ์ให้เป็นระเบียบ",
    tasks: [
      { label: "Home Position", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
      { label: "Pick Up Glass 1", j1: -25, j2: -78, j3: 48, j4: -34, j5: 3, j6: -8, rail: 45, speed: 30, delay: 300, gripper: 80 },
      { label: "Lift Glass 1", j1: -25, j2: -52, j3: 22, j4: -38, j5: 3, j6: -8, rail: 45, speed: 45, delay: 200, gripper: 80 },
      { label: "Place Glass on Tray Left", j1: 30, j2: -80, j3: 50, j4: -36, j5: -3, j6: 5, rail: 160, speed: 20, delay: 400, gripper: 0 },
      { label: "Move to Plate", j1: -40, j2: -58, j3: 28, j4: -38, j5: 0, j6: -15, rail: 90, speed: 65, delay: 100, gripper: 0 },
      { label: "Grip Plate", j1: -40, j2: -82, j3: 52, j4: -32, j5: 0, j6: -15, rail: 90, speed: 20, delay: 500, gripper: 80 },
      { label: "Lift and Transport Plate", j1: 20, j2: -52, j3: 22, j4: -40, j5: 0, j6: 0, rail: 200, speed: 55, delay: 200, gripper: 80 },
      { label: "Place Plate on Tray Center", j1: 20, j2: -82, j3: 52, j4: -36, j5: 0, j6: 0, rail: 200, speed: 20, delay: 400, gripper: 0 },
      { label: "Verify Alignment", j1: 20, j2: -65, j3: 35, j4: -40, j5: 0, j6: 0, rail: 200, speed: 15, delay: 800, gripper: 0 },
      { label: "Adjust Tray Position", j1: 15, j2: -72, j3: 42, j4: -38, j5: 2, j6: -2, rail: 195, speed: 10, delay: 500, gripper: 0 },
      { label: "Return Home", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
    ],
  },
  {
    name: "Beverage Mixing",
    description: "ผสมเครื่องดื่ม: เทส่วนผสม น้ำเชื่อม และนม",
    tasks: [
      { label: "Home Position", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
      { label: "Pick Up Mixing Cup", j1: 38, j2: -75, j3: 45, j4: -32, j5: 6, j6: 12, rail: 75, speed: 35, delay: 300, gripper: 80 },
      { label: "Position Under Syrup Dispenser", j1: 72, j2: -62, j3: 35, j4: -45, j5: 0, j6: 20, rail: 180, speed: 50, delay: 100, gripper: 80 },
      { label: "Dispense Syrup", j1: 72, j2: -75, j3: 48, j4: -40, j5: 0, j6: 20, rail: 180, speed: 15, delay: 2000, gripper: 80 },
      { label: "Move to Milk Station", j1: 108, j2: -62, j3: 35, j4: -45, j5: -5, j6: 35, rail: 300, speed: 55, delay: 100, gripper: 80 },
      { label: "Dispense Milk", j1: 108, j2: -75, j3: 48, j4: -40, j5: -5, j6: 35, rail: 300, speed: 15, delay: 2500, gripper: 80 },
      { label: "Move to Stir Position", j1: 90, j2: -55, j3: 25, j4: -50, j5: 0, j6: 25, rail: 240, speed: 45, delay: 200, gripper: 80 },
      { label: "Stir Beverage", j1: 90, j2: -70, j3: 40, j4: -48, j5: 15, j6: 25, rail: 240, speed: 20, delay: 3000, gripper: 80 },
      { label: "Lift Mixing Cup", j1: 90, j2: -52, j3: 22, j4: -50, j5: 0, j6: 25, rail: 240, speed: 30, delay: 300, gripper: 80 },
      { label: "Place Cup for Serving", j1: 50, j2: -80, j3: 50, j4: -38, j5: 0, j6: 15, rail: 400, speed: 25, delay: 400, gripper: 0 },
      { label: "Return Home", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
    ],
  },
  {
    name: "Topping Application",
    description: "ใส่ท็อปปิ้ง: วิปครีม ซอส และตกแต่งหน้าเครื่องดื่ม",
    tasks: [
      { label: "Home Position", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
      { label: "Grip Whipped Cream Can", j1: -55, j2: -72, j3: 42, j4: -35, j5: -8, j6: -20, rail: 95, speed: 30, delay: 300, gripper: 80 },
      { label: "Position Over Cup", j1: 65, j2: -58, j3: 28, j4: -45, j5: 5, j6: 22, rail: 210, speed: 50, delay: 100, gripper: 80 },
      { label: "Apply Whipped Cream", j1: 65, j2: -68, j3: 38, j4: -42, j5: 5, j6: 22, rail: 210, speed: 10, delay: 2500, gripper: 80 },
      { label: "Return Cream Can", j1: -55, j2: -72, j3: 42, j4: -35, j5: -8, j6: -20, rail: 95, speed: 45, delay: 300, gripper: 0 },
      { label: "Grip Chocolate Sauce", j1: -42, j2: -68, j3: 38, j4: -36, j5: -5, j6: -15, rail: 110, speed: 35, delay: 200, gripper: 80 },
      { label: "Position Over Cup Sauce", j1: 65, j2: -58, j3: 28, j4: -45, j5: 5, j6: 22, rail: 210, speed: 50, delay: 100, gripper: 80 },
      { label: "Drizzle Chocolate Sauce", j1: 65, j2: -65, j3: 35, j4: -42, j5: 18, j6: 22, rail: 210, speed: 8, delay: 1800, gripper: 80 },
      { label: "Return Sauce Bottle", j1: -42, j2: -68, j3: 38, j4: -36, j5: -5, j6: -15, rail: 110, speed: 45, delay: 200, gripper: 0 },
      { label: "Add Garnish", j1: 80, j2: -72, j3: 42, j4: -40, j5: 10, j6: 28, rail: 215, speed: 15, delay: 1200, gripper: 0 },
      { label: "Return Home", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
    ],
  },
  {
    name: "Package Assembly",
    description: "แพ็คสินค้า: ใส่กล่อง ปิดฝา และติดสติกเกอร์",
    tasks: [
      { label: "Home Position", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
      { label: "Pick Up Box Bottom", j1: -48, j2: -78, j3: 48, j4: -34, j5: 0, j6: -18, rail: 115, speed: 30, delay: 300, gripper: 80 },
      { label: "Position Box on Table", j1: 25, j2: -82, j3: 52, j4: -36, j5: 0, j6: 8, rail: 250, speed: 35, delay: 400, gripper: 0 },
      { label: "Grip Product", j1: -30, j2: -75, j3: 45, j4: -35, j5: 5, j6: -10, rail: 70, speed: 25, delay: 400, gripper: 80 },
      { label: "Place Product in Box", j1: 25, j2: -85, j3: 55, j4: -38, j5: 5, j6: 8, rail: 250, speed: 20, delay: 500, gripper: 0 },
      { label: "Align Product", j1: 25, j2: -80, j3: 50, j4: -38, j5: 5, j6: 8, rail: 250, speed: 10, delay: 600, gripper: 0 },
      { label: "Pick Up Box Lid", j1: -55, j2: -72, j3: 42, j4: -35, j5: -5, j6: -22, rail: 130, speed: 30, delay: 300, gripper: 80 },
      { label: "Place Lid on Box", j1: 25, j2: -68, j3: 38, j4: -40, j5: -5, j6: 8, rail: 250, speed: 15, delay: 400, gripper: 0 },
      { label: "Press Lid Down", j1: 25, j2: -78, j3: 48, j4: -38, j5: -5, j6: 8, rail: 250, speed: 10, delay: 800, gripper: 0 },
      { label: "Move Box to Output", j1: 115, j2: -68, j3: 38, j4: -42, j5: 0, j6: 40, rail: 420, speed: 50, delay: 300, gripper: 0 },
      { label: "Return Home", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
    ],
  },
  {
    name: "Station Setup",
    description: "เตรียมสถานี: จัดวางอุปกรณ์และภาชนะก่อนเริ่มงาน",
    tasks: [
      { label: "Home Position", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
      { label: "Pick Up Cup Holder", j1: -68, j2: -75, j3: 45, j4: -35, j5: 2, j6: -25, rail: 55, speed: 35, delay: 300, gripper: 80 },
      { label: "Place Cup Holder Station 1", j1: 45, j2: -78, j3: 48, j4: -36, j5: 2, j6: 15, rail: 150, speed: 25, delay: 400, gripper: 0 },
      { label: "Grab Napkin Holder", j1: -52, j2: -72, j3: 42, j4: -36, j5: -3, j6: -18, rail: 85, speed: 40, delay: 200, gripper: 80 },
      { label: "Place Napkin Holder", j1: 52, j2: -75, j3: 45, j4: -38, j5: -3, j6: 18, rail: 165, speed: 25, delay: 400, gripper: 0 },
      { label: "Arrange Sugar Packets", j1: 18, j2: -80, j3: 50, j4: -38, j5: 8, j6: 5, rail: 130, speed: 15, delay: 1000, gripper: 0 },
      { label: "Position Stir Sticks", j1: -18, j2: -78, j3: 48, j4: -36, j5: -5, j6: -5, rail: 110, speed: 15, delay: 800, gripper: 0 },
      { label: "Check Station Alignment", j1: 0, j2: -55, j3: 25, j4: -42, j5: 0, j6: 0, rail: 140, speed: 20, delay: 1500, gripper: 0 },
      { label: "Fine-tune Positions", j1: 5, j2: -65, j3: 35, j4: -40, j5: 2, j6: 2, rail: 145, speed: 10, delay: 1000, gripper: 0 },
      { label: "Signal Setup Complete", j1: 0, j2: -55, j3: 25, j4: -45, j5: 0, j6: 0, rail: 0, speed: 20, delay: 500, gripper: 0 },
      { label: "Return Home", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
    ],
  },
  {
    name: "Quality Check",
    description: "ตรวจสอบคุณภาพ: สแกนและตรวจสอบสินค้าก่อนส่งมอบ",
    tasks: [
      { label: "Home Position", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
      { label: "Move Camera to Zone A", j1: -30, j2: -50, j3: 20, j4: -40, j5: 0, j6: -10, rail: 80, speed: 60, delay: 200, gripper: 0 },
      { label: "Scan Product Front", j1: -30, j2: -62, j3: 32, j4: -38, j5: 0, j6: -10, rail: 80, speed: 10, delay: 2000, gripper: 0 },
      { label: "Scan Product Side", j1: -15, j2: -62, j3: 32, j4: -40, j5: 12, j6: -5, rail: 80, speed: 10, delay: 1500, gripper: 0 },
      { label: "Scan Product Top", j1: -22, j2: -45, j3: 15, j4: -45, j5: -8, j6: -8, rail: 80, speed: 10, delay: 1500, gripper: 0 },
      { label: "Move Camera to Zone B", j1: 35, j2: -50, j3: 20, j4: -40, j5: 0, j6: 12, rail: 200, speed: 60, delay: 200, gripper: 0 },
      { label: "Inspect Cup Fill Level", j1: 35, j2: -58, j3: 28, j4: -40, j5: 0, j6: 12, rail: 200, speed: 10, delay: 2000, gripper: 0 },
      { label: "Check Lid Seal", j1: 35, j2: -52, j3: 22, j4: -42, j5: -5, j6: 12, rail: 200, speed: 10, delay: 1500, gripper: 0 },
      { label: "Verify Label Placement", j1: 30, j2: -48, j3: 18, j4: -44, j5: 10, j6: 10, rail: 195, speed: 10, delay: 1500, gripper: 0 },
      { label: "Approve and Tag Item", j1: 30, j2: -65, j3: 35, j4: -40, j5: 0, j6: 10, rail: 195, speed: 20, delay: 800, gripper: 0 },
      { label: "Return Home", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
    ],
  },
  {
    name: "Cleanup Workflow",
    description: "ทำความสะอาดสถานี: เก็บอุปกรณ์ เช็ดโต๊ะ และจัดพื้นที่",
    tasks: [
      { label: "Home Position", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
      { label: "Pick Up Cleaning Cloth", j1: -45, j2: -75, j3: 45, j4: -35, j5: 5, j6: -15, rail: 100, speed: 35, delay: 300, gripper: 80 },
      { label: "Wipe Zone A Left Pass", j1: -60, j2: -68, j3: 38, j4: -38, j5: 0, j6: -22, rail: 50, speed: 20, delay: 500, gripper: 80 },
      { label: "Wipe Zone A Right Pass", j1: 0, j2: -68, j3: 38, j4: -38, j5: 0, j6: 0, rail: 50, speed: 20, delay: 500, gripper: 80 },
      { label: "Wipe Zone B Left Pass", j1: -60, j2: -68, j3: 38, j4: -38, j5: 0, j6: -22, rail: 200, speed: 20, delay: 500, gripper: 80 },
      { label: "Wipe Zone B Right Pass", j1: 0, j2: -68, j3: 38, j4: -38, j5: 0, j6: 0, rail: 200, speed: 20, delay: 500, gripper: 80 },
      { label: "Wipe Zone C Pass", j1: -30, j2: -68, j3: 38, j4: -38, j5: 0, j6: -10, rail: 350, speed: 20, delay: 500, gripper: 80 },
      { label: "Return Cleaning Cloth", j1: -45, j2: -75, j3: 45, j4: -35, j5: 5, j6: -15, rail: 100, speed: 40, delay: 300, gripper: 0 },
      { label: "Collect Used Items", j1: 70, j2: -78, j3: 48, j4: -36, j5: -5, j6: 25, rail: 380, speed: 30, delay: 400, gripper: 80 },
      { label: "Deposit Used Items in Bin", j1: 150, j2: -70, j3: 40, j4: -40, j5: -5, j6: 55, rail: 460, speed: 35, delay: 500, gripper: 0 },
      { label: "Final Return Home", j1: 0, j2: -45, j3: 0, j4: -45, j5: 0, j6: 0, rail: 0, speed: 80, delay: 200, gripper: 0 },
    ],
  },
];

async function main() {
  console.log("🌱 Seeding database with realistic café robot sequences...");

  // Clear existing data
  await prisma.task.deleteMany();
  await prisma.job.deleteMany();

  for (let jobIdx = 0; jobIdx < jobSequences.length; jobIdx++) {
    const jobDef = jobSequences[jobIdx];
    const job = await prisma.job.create({
      data: {
        name: jobDef.name,
        description: jobDef.description,
        tasks: {
          create: jobDef.tasks.map((t, taskIdx) => ({
            sequence: taskIdx + 1,
            label: t.label,
            j1: t.j1,
            j2: t.j2,
            j3: t.j3,
            j4: t.j4,
            j5: t.j5,
            j6: t.j6,
            rail: t.rail,
            speed: t.speed,
            delay: t.delay,
            gripper: t.gripper,
          })),
        },
      },
      include: { tasks: true },
    });

    console.log(
      `✅ Created Job #${jobIdx + 1}: "${job.name}" with ${job.tasks.length} tasks`
    );
  }

  console.log("🎉 Seeding complete! Realistic café robot sequences ready.");
}

main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (e) => {
    console.error("❌ Seeding failed:", e);
    await prisma.$disconnect();
    process.exit(1);
  });
