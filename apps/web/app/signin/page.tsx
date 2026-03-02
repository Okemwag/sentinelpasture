import Link from "next/link";

export default function SignInPage() {
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

          <form className="space-y-4">
            <div>
              <label
                htmlFor="email"
                className="block text-[13px] text-[#6B7280] mb-2"
              >
                Email Address
              </label>
              <input
                type="email"
                id="email"
                className="w-full px-3 py-2 text-[15px] border border-[#E5E7EB] rounded-[8px] bg-white text-[#111111] focus:outline-none focus:border-[#374151]"
                placeholder="user@example.gov"
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
                className="w-full px-3 py-2 text-[15px] border border-[#E5E7EB] rounded-[8px] bg-white text-[#111111] focus:outline-none focus:border-[#374151]"
                placeholder="••••••••"
              />
            </div>

            <div className="pt-2">
              <Link
                href="/dashboard"
                className="block w-full px-4 py-3 text-[15px] text-center bg-[#374151] text-white rounded-[8px] hover:bg-[#1F2937] transition-colors duration-150"
              >
                Sign In
              </Link>
            </div>
          </form>

          <div className="mt-6 pt-6 border-t border-[#E5E7EB]">
            <p className="text-[13px] text-[#6B7280] text-center">
              Authorized personnel only. All access is logged and monitored.
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
