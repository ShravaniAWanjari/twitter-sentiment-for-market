# End-to-End Deployment Plan

This document outlines the deployment strategy for the Capstone Research Platform. The architecture consists of a static **Vite frontend** and a heavy-duty **FastAPI backend** running deep learning models (PyTorch/Transformers).

Because the backend relies on custom-trained model weights (stored locally in `experiments/runs/`) and requires significant RAM/compute for inference, we have to handle the model artifacts and choose the right compute environment carefully.

Below are two main deployment options: **The Completely Free Route** and **The AWS EC2 Route (Small Charge but Better Performance)**.

---

## Architecture Overview
1. **Frontend**: Vite Application (served statically).
2. **Backend**: FastAPI Application (Python).
3. **Artifacts Management**: The custom model checkpoints (`experiments/runs/...`) will need to be pushed to an online storage (like AWS S3 or Hugging Face Models hub) so the deployment server can download them.

---

## Option 1: The "Small AWS Charge" Route (Recommended for Speed)
*Estimated Cost: ~$0.52/hour for a T4 GPU instance, or ~$0.08/hour for a decent CPU instance.*

For a demo day or short-term usage (~48 hours), using AWS EC2 is the most robust way to ensure your model runs quickly without memory crashes. 

### Step 1: Host the Frontend on Vercel or Netlify (Free)
1. Push your `frontend/` code to a GitHub repository.
2. Sign up for [Vercel](https://vercel.com/) (Free).
3. Import the GitHub repository. Vercel automatically detects Vite.
4. Set the build command to `npm run build` and output directory to `dist`.
5. Add an Environment Variable for your backend API (e.g., `VITE_API_URL = http://<YOUR_EC2_IP>:8000`).
6. Deploy.

### Step 2: Upload Model Artifacts to Cloud Storage (AWS S3)
1. In AWS, create an S3 bucket (e.g., `capstone-models-storage`).
2. Upload the `experiments/runs/` folder to this bucket. Alternatively, you can use **GitHub LFS**, but S3 is easier for large checkpoints.

### Step 3: Spin Up an AWS EC2 Instance for the Backend
1. Go to AWS EC2 and launch a new instance.
   - **Performance (No GPU)**: `t3.xlarge` (4 vCPUs, 16 GB RAM) - ~$0.16/hr. Good for moderate speed.
   - **Performance (GPU - Best)**: `g4dn.xlarge` (4 vCPUs, 16 GB RAM, 1 T4 GPU) - ~$0.52/hr. Lightning fast analysis.
   - **OS**: Use the **"Deep Learning AMI GPU PyTorch"** (Ubuntu 20.04 or 22.04). It has PyTorch and CUDA pre-installed.
2. Configure Security Group:
   - Allow **TCP port 8000** (for your FastAPI) from Anywhere (`0.0.0.0/0`).
   - Allow **SSH port 22** for you to remote in.
3. SSH into your EC2 instance once launched.

### Step 4: Run the Backend on EC2
1. Clone your project repository onto the EC2 instance.
2. Download your model weights from S3 into the `experiments/runs/` directory using the AWS CLI.
   ```bash
   aws s3 cp s3://capstone-models-storage/experiments/ ./experiments/ --recursive
   ```
3. Create a Python virtual environment and install the requirements:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r backend/requirements.txt
   ```
4. Start the FastAPI server using `uvicorn` (with `nohup` or `tmux` to keep it running after you close the terminal):
   ```bash
   nohup uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
   ```
5. *Done!* Your frontend on Vercel can now hit `http://<EC2-IP>:8000`.

---

## Option 2: The Completely Free Route
*Estimated Cost: $0.00.*

If you want to avoid giving AWS your credit card entirely, you can host the heavy backend on **Hugging Face Spaces** using their free Docker tier.

### Step 1: Manage Frontend
- Same as above: Deploy the Vite frontend to **Vercel** for free. Point the `VITE_API_URL` to your future Hugging Face Space URL.

### Step 2: Setup Hugging Face Space for the Backend
1. Create an account on Hugging Face (Free).
2. Create a new **Space** and select **Docker** as the SDK. You get a free tier of 16 GB RAM and 2 vCPUs.
3. Your local weights are too large for standard Git, so you'll push the backend code and weights to the Space using Git LFS (Large File Storage).

### Step 3: Prepare the Dockerfile
You will add a `Dockerfile` to your backend folder that tells Hugging Face how to run it:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, models, and data
COPY backend/ /app/backend/
COPY core/ /app/core/
COPY experiments/ /app/experiments/
COPY .env /app/.env

# Run uvicorn on port 7860 (HF Spaces expects 7860)
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Step 4: Push to Hugging Face
1. Initialize a git repo inside your Space clone, install `git-lfs` to track all your `.bin`, `.pt`, and `.safetensors` files in `experiments/runs/`.
2. Push everything to the Hugging Face remote repository.
3. The Space will automatically build the Docker image and start the FastAPI server.
4. *Limits of Free Tier*: Inference might take 2–5 seconds per headline since it lacks a GPU. The Space will go to "Sleep" if no one uses it for 48 hours, but wakes up on the next request.

---

## Summary of Action Plan (Next Steps)

1. **Decide on the tier:** Choose whether $10-15 total for the weekend is worth having a fast, dedicated EC2 server without the headache of Git LFS and Docker configuration. **(AWS is recommended for a smooth Capstone presentation).**
2. **Update the Frontend:** Modify the frontend code so that API calls are not hardcoded to `localhost:8000` but instead use an environment variable (e.g. `import.meta.env.VITE_API_URL`).
3. **Upload Artifacts:** Move the heavy directories (`dataset/`, `experiments/runs/`) to S3 if going the AWS route.
4. **Deploy:** Spin up Vercel + AWS EC2, link them together, and showcase!
