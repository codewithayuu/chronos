import "./globals.css";

export const metadata = {
  title: "Project Iris Dashboard",
  description: "The Many-Eyed Messenger",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased text-white bg-slate-950">{children}</body>
    </html>
  );
}
