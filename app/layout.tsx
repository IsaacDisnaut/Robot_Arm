import type { Metadata } from "next";
import "./globals.css";
import { RosProvider } from "@/context/RosContext";
import SafetyOverlay from "@/components/SafetyOverlay";

export const metadata: Metadata = {
  title: "Robot Arm Studio",
  description: "Control robot arm via ROS",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className="antialiased h-full w-full m-0 p-0 overflow-hidden">
        <RosProvider>
          <SafetyOverlay />
          {children}
        </RosProvider>
      </body>
    </html>
  );
}
