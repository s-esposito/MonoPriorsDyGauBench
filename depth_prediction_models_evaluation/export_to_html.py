import os
import json
import base64
import numpy as np
from glob import glob
from PIL import Image

def encode_file_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_npy_base64(path):
    data = np.load(path)
    return base64.b64encode(data.tobytes()).decode("utf-8"), list(data.shape)

def load_camera_json(json_path):
    with open(json_path, "r") as f:
        cam = json.load(f)

    R = np.array(cam["orientation"])  # 3x3
    t = np.array(cam["position"]).reshape(3, 1)  # 3x1

    T_wc = np.eye(4)
    T_wc[:3, :3] = R
    T_wc[:3, 3:] = t
    T_cw = np.linalg.inv(T_wc)  # Invert to get camera-to-world
    pose = T_cw.astype(np.float32)

    fx = fy = cam["focal_length"]
    cx, cy = cam["principal_point"]
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0, 1]
    ], dtype=np.float32)

    return pose, K


def create_scene_data(rgb_dir, depth_dir, camera_dir, max_frames=100):
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith((".png", ".jpg"))])[:max_frames]
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npy")])[:max_frames]
    camera_files = sorted([f for f in os.listdir(camera_dir) if f.endswith(".json")])[:max_frames]

    scene = {"frames": []}
    for rgb_file, depth_file, cam_file in zip(rgb_files, depth_files, camera_files):
        rgb_path = os.path.join(rgb_dir, rgb_file)
        depth_path = os.path.join(depth_dir, depth_file)
        cam_path = os.path.join(camera_dir, cam_file)

        pose, K = load_camera_json(cam_path)
        depth_b64, depth_shape = load_npy_base64(depth_path)

        frame = {
            "image": encode_file_base64(rgb_path),
            "depth": depth_b64,
            "depth_shape": depth_shape,
            "intrinsics": K.flatten().tolist(),
            "pose": pose.flatten().tolist()
        }
        scene["frames"].append(frame)

    return scene


def embed_scene_to_html(scene_data, output_path="viewer.html"):
    json_data = json.dumps(scene_data)
    b64_data = base64.b64encode(json_data.encode()).decode()

    with open(output_path, "w") as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>3D Viewer</title>
    <style>body {{ margin: 0; overflow: hidden; }}</style>
</head>
<body>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script>
window.embeddedBase64 = "{b64_data}";
</script>
<script>
(async function() {{
    const blob = atob(window.embeddedBase64);
    const json = JSON.parse(blob);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(70, window.innerWidth/window.innerHeight, 0.01, 100);
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const frames = json.frames;

    for (let frame of frames) {{
        const img = new Image();
        img.src = "data:image/png;base64," + frame.image;
        await img.decode();

        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const imgData = ctx.getImageData(0, 0, img.width, img.height).data;

        const depth_bytes = atob(frame.depth);
        const depth = new Float32Array(new Uint8Array([...depth_bytes].map(c => c.charCodeAt(0))).buffer);
        const [H, W] = frame.depth_shape;

        const fx = frame.intrinsics[0], fy = frame.intrinsics[4];
        const cx = frame.intrinsics[2], cy = frame.intrinsics[5];
        const pose = new THREE.Matrix4().fromArray(frame.pose);

        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];

        for (let i = 0; i < H; i += 2) {{
            for (let j = 0; j < W; j += 2) {{
                const idx = i * W + j;
                const z = depth[idx];
                if (z <= 0 || z > 10) continue;

                const x = (j - cx) * z / fx;
                const y = (i - cy) * z / fy;
                const pt = new THREE.Vector3(x, y, z).applyMatrix4(pose);

                positions.push(pt.x, pt.y, pt.z);

                const r = imgData[4 * idx] / 255;
                const g = imgData[4 * idx + 1] / 255;
                const b = imgData[4 * idx + 2] / 255;
                colors.push(r, g, b);
            }}
        }}

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({{ size: 0.01, vertexColors: true }});
        const cloud = new THREE.Points(geometry, material);
        scene.add(cloud);
    }}

    camera.position.z = 2;
    function animate() {{
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }}
    animate();
}})();
</script>
</body>
</html>
        """)
    print(f"✅ Saved viewer to {output_path}")
    
if __name__ == "__main__":
    rgb_dir = "/home/geiger/gwb215/datasets/iphone/sriracha-tree/rgb/1x"
    depth_dir = "/home/geiger/gwb215/datasets/iphone/sriracha-tree/flow3d_preprocessed/video_depth_anything/1x"
    camera_dir = "/home/geiger/gwb215/datasets/iphone/sriracha-tree/camera"
    scene_data = create_scene_data(rgb_dir, depth_dir, camera_dir, max_frames=10)
    embed_scene_to_html(scene_data, output_path="viewer.html")