from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import io
import tempfile
import os
import time
from gradio_client import Client, handle_file

app = FastAPI()

_gradio_client = None
_fallback_client = None

def get_gradio_client(use_fallback=False):
    global _gradio_client, _fallback_client
    if use_fallback:
        if _fallback_client is None:
            print("Connecting to fallback space (leonelhs/CodeFormer)...")
            _fallback_client = Client("leonelhs/CodeFormer")
        return _fallback_client
    else:
        if _gradio_client is None:
            print("Connecting to primary space (sczhou/CodeFormer)...")
            _gradio_client = Client("sczhou/CodeFormer")
        return _gradio_client

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
        
        result = None
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
            except Exception as err2:
                print(f"Primary space failed (Attempt 2): {str(err2)}")
                last_err = err2
                
                # Attempt 3: Switch to fallback duplicate space
                try:
                    print("Attempt 3: Trying fallback space (leonelhs/CodeFormer)...")
                    client = get_gradio_client(use_fallback=True)
                    result = client.predict(
                        image=handle_file(temp_path),
                        face_align=face_align,
                        background_enhance=True,
                        face_upsample=face_upsample,
                        upscale=4.0,
                        codeformer_fidelity=0.6,
                        api_name="/inference"
                    )
                except Exception as err3:
                    print(f"Fallback space failed: {str(err3)}")
                    last_err = err3

        if result is None:
            # Raise the final error encountered
            raise last_err
            
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
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Upscaling failed. The AI spaces are currently busy: {str(e)}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
