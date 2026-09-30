# Frontend Handoff: Multi-Face Attendance

## Backend Endpoint

Use the FastAPI endpoint:

```text
POST /api/v1/appside/attendance/mark-multiple
```

The request must contain exactly five camera frames in a multipart form field named `images`.

Optional form fields:

- `mode`: `check_in` or `check_out`
- `action`: `check_in` or `check_out`
- `latitude`: number
- `longitude`: number
- `location_name`: string

Example browser request:

```javascript
const formData = new FormData();
frames.forEach((blob, index) => {
  formData.append('images', blob, `frame-${index + 1}.jpg`);
});
formData.append('mode', 'check_in');

const response = await fetch(
  `${API_URL}/api/v1/appside/attendance/mark-multiple`,
  {
    method: 'POST',
    body: formData,
  }
);

const result = await response.json();
```

## Processing Behavior

The backend:

1. Receives five frames.
2. Detects all faces in each frame.
3. Tracks faces across the five frames.
4. Matches each detected face against enrolled employees.
5. Accepts the closest employee match when its confidence is at least `0.60`.
6. Marks attendance separately for every recognized employee.
7. Deduplicates the same employee so one employee is not marked twice in one batch.

Unknown faces are never marked for attendance.

Recognition uses cosine similarity against every enrolled employee embedding. The
closest match is selected, but it is accepted only when its score is at least
`0.60`. Scores below `0.60` are returned and displayed as `Unknown`.

## Response Shape

A recognized employee is returned in `processed_employees`:

```json
{
  "employee_id": "001",
  "employee_name": "Employee Name",
  "success": true,
  "action": "check_in",
  "message": "Check-in successful.",
  "confidence": 0.783,
  "recognition_confidence": 0.783,
  "liveness_average": 0.91,
  "live_frames": 5,
  "total_frames": 5,
  "attendance_id": 12,
  "check_in": "09:30:00",
  "check_out": null,
  "working_minutes": null,
  "status": "present"
}
```

An unknown face is also returned in `processed_employees`:

```json
{
  "employee_id": null,
  "employee_name": "Unknown",
  "success": false,
  "action": null,
  "message": "Unknown face.",
  "status": "unknown",
  "confidence": 0.19,
  "recognition_confidence": 0.19,
  "liveness_average": 0.84,
  "live_frames": 5,
  "total_frames": 5,
  "attendance_id": null,
  "check_in": null,
  "check_out": null,
  "working_minutes": null
}
```

A single response can contain multiple recognized employees and multiple unknown faces:

```javascript
for (const person of result.processed_employees || []) {
  if (person.employee_name === 'Unknown' || person.status === 'unknown') {
    renderUnknownFace(person.confidence);
    continue;
  }

  renderEmployeeResult({
    employeeCode: person.employee_id,
    name: person.employee_name,
    confidence: person.confidence,
    attendanceMarked: person.success,
    message: person.message,
  });
}
```

## Frontend Display Requirements

For every result, display:

- Employee code from `employee_id` for recognized people.
- `Unknown` when `employee_name` is `Unknown`, `status` is `unknown`, or `employee_id` is null.
- Confidence from `confidence` or `recognition_confidence`.
- Attendance result from `success`, `action`, `status`, and `message`.
- Liveness information from `live_frames`, `total_frames`, and `liveness_average` when useful.

Confidence is a decimal from `0` to `1`. Convert it to a percentage only when the UI explicitly uses percentage formatting:

```javascript
const percentage = `${(person.confidence * 100).toFixed(1)}%`;
```

For the camera overlay, the desired visual labels are:

```text
Employee: 001
Score: 0.783
```

and for an unmatched live face:

```text
Unknown
Score: 0.190
```

## Important Frontend Notes

- Do not assume only one person exists in the response.
- Do not stop rendering after the first recognized employee.
- Do not send fewer or more than five images to `mark-multiple`.
- Unknown records should be displayed but must not show a successful attendance state.
- A response with only unknown faces has `success: false` and still contains the unknown records in `processed_employees`.
- The existing `/demo` page already renders result cards and confidence values after the API response. It does not currently draw live bounding boxes over the video.
- The OpenCV labels in `recognize.py` are a separate desktop display and are not automatically visible in the browser frontend.

## Local Verification

Start the backend:

```powershell
python -m uvicorn app.main:app --reload
```

Open the browser demo:

```text
http://localhost:8000/demo
```

Use two enrolled faces in the camera view and capture five frames. The result list should contain one result per recognized employee, plus an `Unknown` result for every unmatched live face.
