import os
import json

path = '/home/isaac/ros2_ws/src/robot_arm/tasks/data.txt'

# 1. อ่านข้อมูลเดิมเพื่อหา Task ล่าสุด
existing_data = []
task = 1

if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            existing_data = json.load(f)
            if len(existing_data) > 0:
                last_task = existing_data[-1]["task_no"]
                task = last_task + 1
        except json.JSONDecodeError:
            pass # ถ้าไฟล์ว่างให้เริ่มที่ 1

# ตรวจสอบว่าไม่เกิน 10 task
if len(existing_data) >= 10:
    print(f"❌ ไม่สามารถเพิ่ม Task ใหม่ได้: ข้อมูลเต็มความจุแล้ว (สูงสุด 10 Task)")
else:
    # ---------------------------------------------------------
    # 2. เตรียมข้อมูลส่วนของ Jog โดยใช้ตัวแปร iteration
    # ---------------------------------------------------------
    iteration = 2  # กำหนด n ได้ตามต้องการ (ตัวอย่างนี้ n=3 จะสร้างถึง jog3)
    
    # ข้อมูลเริ่มต้น (Base Jog) ที่จะถูกนำไปบวกเพิ่มตามหมายเลข jog
    base_jog = [0,0,0] 

    # สร้าง Dictionary สำหรับ Task ใหม่
    new_task = {"task_no": task}
    
    # วนลูปตั้งแต่รอบที่ 1 ถึง iteration (เช่น 1, 2, 3)
    for i in range(1, iteration + 1):
        
        # เอาค่า q ใน base_jog มาบวกด้วย 'หมายเลข jog (i)'
        jog = [round(q + i, 1) for q in base_jog] 
        
        # ตั้งชื่อ Key เป็น jog1, jog2, ... และจับคู่กับค่าที่บวกแล้ว
        key_name = f"jog{i}"
        new_task[key_name] = jog
        
        print(f"ประมวลผล {key_name}: บวกเพิ่มไป {i} -> {jog}")

    # นำ Task ใหม่ไปต่อท้ายข้อมูลเดิม
    existing_data.append(new_task)

    # 3. บันทึกข้อมูลทั้งหมดกลับลงไฟล์
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    print(f"\n✅ บันทึกข้อมูล Task ที่ {task} (สร้างทั้งหมด {iteration} jogs) เรียบร้อยแล้ว!")

# 4. ทดสอบอ่านไฟล์เพื่อแสดงผล
try:
    with open(path, 'r', encoding='utf-8') as f:
        print("\n--- ข้อมูลในไฟล์ ---")
        print(f.read())
except FileNotFoundError:
    print("ไม่พบไฟล์")