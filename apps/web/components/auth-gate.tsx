"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";

import { clearAuthSession, readAuthSession } from "@/lib/auth-session";
import { apiClient } from "@/lib/api-client";

export default function AuthGate({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [status, setStatus] = useState<"checking" | "ready">("checking");

  useEffect(() => {
    let isActive = true;

    async function verify() {
      const session = readAuthSession();
      if (!session) {
        router.replace(`/signin?next=${encodeURIComponent(pathname)}`);
        return;
      }

      try {
        await apiClient.getCurrentUser();
        if (isActive) {
          setStatus("ready");
        }
      } catch {
        clearAuthSession();
        router.replace(`/signin?next=${encodeURIComponent(pathname)}`);
      }
    }

    void verify();

    return () => {
      isActive = false;
    };
  }, [pathname, router]);

  if (status !== "ready") {
    return (
      <div className="flex min-h-screen items-center justify-center text-sm text-[#6B7280]">
        Verifying access...
      </div>
    );
  }

  return <>{children}</>;
}
