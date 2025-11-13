import React, { useState } from "react";

const severityColors = {
  "Нет": "bg-gray-400",
  "Слабый": "bg-green-500",
  "Средний": "bg-yellow-400",
  "Сильный": "bg-red-600",
};

const categories = [
  "Sex & Nudity",
  "Violence & Gore",
  "Profanity",
  "Alcohol, Drugs & Smoking",
  "Frightening & Intense Scenes",
];

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [openCategory, setOpenCategory] = useState(null); // какой блок открыт

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      const taskId = data.task_id;

      // Polling
      let taskResult = null;
      while (!taskResult) {
        await new Promise((r) => setTimeout(r, 1000));
        const resultRes = await fetch(`http://localhost:8000/result/${taskId}`);
        const resultData = await resultRes.json();

        if (resultData.status === "success") {
          taskResult = resultData.result;
        } else if (resultData.status === "failed") {
          alert("Ошибка анализа: " + resultData.error);
          break;
        }
      }

      if (taskResult) setResult(taskResult);
    } catch (e) {
      console.error(e);
      alert("Ошибка загрузки файла");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <h1 className="text-3xl font-bold text-center mb-4">Age Rating Analyzer</h1>

      <div className="flex space-x-2 items-center">
        <input type="file" onChange={handleFileChange} className="border p-2 rounded" />
        <button
          onClick={handleUpload}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
          disabled={loading}
        >
          {loading ? "Загрузка..." : "Анализировать"}
        </button>
      </div>

      <div className="border-t pt-4 space-y-4">
        <div className="text-lg font-semibold">
          AgeCategory: <span className="text-blue-700">{result?.AgeCategory || "-"}</span>
        </div>

        {categories.map((cat) => {
          const severity = result?.ParentsGuide?.[cat]?.Severity || "Нет";
          const summary = result?.ParentsGuide?.[cat]?.Reason || "Нет данных";
          const isOpen = openCategory === cat;

          return (
            <div key={cat} className="border rounded p-2">
              <div
                className="flex items-center space-x-2 cursor-pointer"
                onClick={() => setOpenCategory(isOpen ? null : cat)}
              >
                <div className={`w-4 h-8 ${severityColors[severity] || "bg-gray-400"} rounded`}></div>
                <span className="font-medium">{cat}: {severity}</span>
                <span className="ml-auto">{isOpen ? "▲" : "▼"}</span>
              </div>

              {isOpen && (
                <div className="mt-2 p-2 bg-gray-100 rounded">
                  {summary}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
