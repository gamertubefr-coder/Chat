
module.exports = {
    daemon: true,
    run: [
        {
            method: "shell.run",
            params: {
                venv: "venv",
                path: "app",
                env: {
                    // Ensure libs are in path for runtime if needed (though static linking preferred or dlls)
                },
                message: [
                    // Compile C++ app if not exists or if source is newer
                    "g++ ../src/main.cpp -o ../main.exe -static -lcomdlg32 -lgdi32 -lwinmm -lgdiplus -lshlwapi",
                    // Start API server
                    "python api.py"
                ],
                on: [{
                    // Wait for API to start
                    "event": "/(Application startup complete)/",
                    "done": true
                }]
            }
        },
        {
            // Run the C++ App
            method: "shell.run",
            params: {
                path: ".",
                message: [
                    "main.exe"
                ]
            }
        }
    ]
}
