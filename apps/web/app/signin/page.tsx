"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

import { storeAuthSession } from "@/lib/auth-session";
import { apiClient } from "@/lib/api-client";

export default function SignInPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsSubmitting(true);
    try {
      const session = await apiClient.login(username, password);
      storeAuthSession(session);
      router.push("/dashboard");
    } catch (submitError) {
      setError(
        submitError instanceof Error ? submitError.message : "Unable to sign in",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#FAFAFA] flex items-center justify-center p-6">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-[8px] border border-[#E5E7EB] p-8">
          <div className="mb-8">
            <h1 className="text-[28px] font-semibold text-[#111111] mb-2">
              Sign In
            </h1>
            <p className="text-[15px] text-[#6B7280]">
              National Risk Intelligence & Resilience Platform
            </p>
          </div>

          <form className="space-y-4" onSubmit={handleSubmit}>
            <div>
              <label
                htmlFor="username"
                className="block text-[13px] text-[#6B7280] mb-2"
              >
                Username
              </label>
              <input
                type="text"
                id="username"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                className="w-full px-3 py-2 text-[15px] border border-[#E5E7EB] rounded-[8px] bg-white text-[#111111] focus:outline-none focus:border-[#374151]"
                placeholder="analyst"
                autoComplete="username"
                required
              />
            </div>

            <div>
              <label
                htmlFor="password"
                className="block text-[13px] text-[#6B7280] mb-2"
              >
                Password
              </label>
              <input
                type="password"
                id="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className="w-full px-3 py-2 text-[15px] border border-[#E5E7EB] rounded-[8px] bg-white text-[#111111] focus:outline-none focus:border-[#374151]"
                placeholder="••••••••"
                autoComplete="current-password"
                required
              />
            </div>

            {error ? (
              <p className="text-[13px] text-[#B91C1C]">{error}</p>
            ) : null}

            <div className="pt-2">
              <button
                type="submit"
                disabled={isSubmitting}
                className="block w-full px-4 py-3 text-[15px] text-center bg-[#374151] text-white rounded-[8px] hover:bg-[#1F2937] transition-colors duration-150"
              >
                {isSubmitting ? "Signing In..." : "Sign In"}
              </button>
            </div>
          </form>

          <div className="mt-6 pt-6 border-t border-[#E5E7EB]">
            <p className="text-[13px] text-[#6B7280] text-center">
              Authorized personnel only. All access is logged and audited.
            </p>
          </div>
        </div>

        <div className="mt-6 text-center">
          <Link
            href="/"
            className="text-[13px] text-[#374151] hover:underline"
          >
            ← Back to home
          </Link>
        </div>
      </div>
    </div>
  );
}
