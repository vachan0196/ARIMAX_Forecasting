# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the requirements file into the container at /app
COPY requirements.txt ./

# Step 4: Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your application code to the container
COPY . .

# Step 6: Expose the port the app runs on
EXPOSE 8050

# Step 7: Define the command to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server"]
