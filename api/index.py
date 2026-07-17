from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io
import tempfile
import os
import time
import urllib.request
from gradio_client import Client, handle_file

app = FastAPI()

_gradio_client = None
_fallback_client = None

def get_gradio_client(use_fallback=False):
    global _gradio_client, _fallback_client
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_token = hf_token.strip()
        
    if use_fallback:
        if _fallback_client is None:
            print("Connecting to fallback space (leonelhs/CodeFormer)...")
            _fallback_client = Client("leonelhs/CodeFormer", token=hf_token)
        return _fallback_client
    else:
        if _gradio_client is None:
            print("Connecting to primary space (sczhou/CodeFormer)...")
            _gradio_client = Client("sczhou/CodeFormer", token=hf_token)
        return _gradio_client

@app.get("/api/debug-image")
async def debug_image():
    temp_path = None
    try:
        # Download a low-res face image from Hugging Face Space inputs
        low_res_url = "https://huggingface.co/spaces/sczhou/CodeFormer/resolve/main/CodeFormer/assets/restoration_result1.png"
        temp_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        print(f"Downloading test face image from {low_res_url}...")
        urllib.request.urlretrieve(low_res_url, temp_path)
        
        client = get_gradio_client(use_fallback=False)
        print("Calling CodeFormer API with upscale=4.0 for test image...")
        result = client.predict(
            image=handle_file(temp_path),
            face_align=True,
            background_enhance=True,
            face_upsample=True,
            upscale=4.0,
            codeformer_fidelity=0.6,
            api_name="/inference"
        )
        output_img_path = result[0]
        with open(output_img_path, "rb") as f:
            output_bytes = f.read()
            
        try:
            os.unlink(output_img_path)
        except:
            pass
            
        return StreamingResponse(io.BytesIO(output_bytes), media_type="image/png")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.get("/api/debug")
async def debug():
    logs = []
    try:
        logs.append("Starting debug test...")
        logs.append(f"HF_TOKEN present: {os.environ.get('HF_TOKEN') is not None}")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            from PIL import Image
            img = Image.new("RGB", (100, 100), color="white")
            img.save(f.name)
            temp_path = f.name
        
        logs.append(f"Created temp image: {temp_path}")
        
        logs.append("Connecting to sczhou/CodeFormer...")
        client = get_gradio_client(use_fallback=False)
        
        logs.append("Calling predict on sczhou/CodeFormer...")
        result = client.predict(
            image=handle_file(temp_path),
            face_align=True,
            background_enhance=True,
            face_upsample=True,
            upscale=4.0,
            codeformer_fidelity=0.6,
            api_name="/inference"
        )
        logs.append(f"SUCCESS! Result: {str(result)}")
        return {"status": "success", "logs": logs}
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        logs.append(f"FAILED: {str(e)}")
        logs.append(err_msg)
        return JSONResponse(status_code=500, content={"status": "error", "logs": logs})

@app.post("/api/upscale")
async def upscale(
    file: UploadFile = File(...),
    model_id: str = Form("codeformer-ultra-4x")
):
    image_bytes = await file.read()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name
        
        face_upsample = True if model_id == "codeformer-ultra-4x" else False
        face_align = True if model_id == "codeformer-ultra-4x" else False
        
        output_img_path = None
        last_err = None
        
        # Attempt 1: Call primary space
        try:
            print("Attempt 1: Calling primary space (sczhou/CodeFormer)...")
            client = get_gradio_client(use_fallback=False)
            result = client.predict(
                image=handle_file(temp_path),
                face_align=face_align,
                background_enhance=True,
                face_upsample=face_upsample,
                upscale=4.0,
                codeformer_fidelity=0.6,
                api_name="/inference"
            )
            output_img_path = result[0]
        except Exception as err1:
            print(f"Primary space failed (Attempt 1): {str(err1)}")
            last_err = err1
            
            # Attempt 2: Pause, clear connection cache, and retry primary space
            time.sleep(2.0)
            try:
                print("Attempt 2: Retrying primary space after cache reset...")
                global _gradio_client
                _gradio_client = None  # Reset client connection state
                client = get_gradio_client(use_fallback=False)
                result = client.predict(
                    image=handle_file(temp_path),
                    face_align=face_align,
                    background_enhance=True,
                    face_upsample=face_upsample,
                    upscale=4.0,
                    codeformer_fidelity=0.6,
                    api_name="/inference"
                )
                output_img_path = result[0]
            except Exception as err2:
                print(f"Primary space failed (Attempt 2): {str(err2)}")
                last_err = err2
                
                # Attempt 3: Switch to fallback duplicate space with its specific /predict endpoint
                try:
                    print("Attempt 3: Trying fallback space (leonelhs/CodeFormer)...")
                    client = get_gradio_client(use_fallback=True)
                    result = client.predict(
                        image=handle_file(temp_path),
                        api_name="/predict"
                    )
                    output_img_path = result[1]
                except Exception as err3:
                    print(f"Fallback space failed: {str(err3)}")
                    last_err = err3

        if output_img_path is None:
            import traceback
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=500,
                detail=f"All attempts failed.\nPrimary Space Error: {str(last_err)}\nTraceback: {tb}"
            )
            
        with open(output_img_path, "rb") as f:
            output_bytes = f.read()
            
        try:
            os.unlink(output_img_path)
        except:
            pass
            
        return StreamingResponse(io.BytesIO(output_bytes), media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Upscaling failed: {str(e)}\nTraceback: {tb}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
