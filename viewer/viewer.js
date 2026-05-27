import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// --- GLB path (query param with default) ---

const params = new URLSearchParams(window.location.search);
const runId = params.get('run') || '20260527_165342';
const glbPath = `../data/runs/${runId}/body_mesh.glb`;

// --- Canvas & renderer ---

const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

// --- Scene ---

const scene = new THREE.Scene();

// --- Camera ---

const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 20);

// --- Controls (created after mesh loads so we have a target) ---

let controls = null;

// --- Lighting: three-point + ambient ---

const ambient = new THREE.AmbientLight(0xffffff, 0.18);
scene.add(ambient);

const keyLight = new THREE.DirectionalLight(0xfff5ee, 4.5);
keyLight.position.set(2.5, 2.0, 2.0);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xd8dde8, 1.2);
fillLight.position.set(-1.8, 0.7, 1.5);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xffffff, 4);
rimLight.position.set(0, 2.0, -2.5);
scene.add(rimLight);

// --- Ground shadow texture (canvas-generated radial gradient) ---

function makeShadowTexture() {
  const size = 256;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');

  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0.0, 'rgba(0,0,0,0.45)');
  gradient.addColorStop(0.3, 'rgba(0,0,0,0.15)');
  gradient.addColorStop(0.7, 'rgba(0,0,0,0.02)');
  gradient.addColorStop(1.0, 'rgba(0,0,0,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  return tex;
}

// --- Load mesh ---

const loader = new GLTFLoader();
loader.load(
  glbPath,
  (gltf) => {
    const mesh = gltf.scene;

    mesh.traverse((child) => {
      if (child.isMesh) {
        child.geometry.computeVertexNormals();
        child.material = new THREE.MeshStandardMaterial({
          color: 0xe8dcc8,
          roughness: 0.65,
          metalness: 0.0,
          flatShading: false,
        });
      }
    });

    scene.add(mesh);

    // Bounding box
    const box = new THREE.Box3().setFromObject(mesh);
    const center = new THREE.Vector3();
    box.getCenter(center);
    const size = new THREE.Vector3();
    box.getSize(size);

    // Camera distance
    const margin = 1.4;
    const aspect = window.innerWidth / window.innerHeight;
    const halfFov = (camera.fov / 2) * (Math.PI / 180);
    const dV = (size.y / 2) / Math.tan(halfFov) * margin;
    const dH = (size.x / 2) / (Math.tan(halfFov) * aspect) * margin;
    const distance = Math.max(dV, dH);

    const elevation = size.y * 0.08;
    camera.position.set(center.x, center.y + elevation, center.z + distance);
    camera.lookAt(center);

    // OrbitControls
    controls = new OrbitControls(camera, canvas);
    controls.target.copy(center);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = distance * 0.3;
    controls.maxDistance = distance * 3.0;
    controls.maxPolarAngle = Math.PI / 2 + 0.35;
    controls.update();

    // Ground plane
    const shadowRadius = size.x * 0.8;
    const shadowGeo = new THREE.PlaneGeometry(shadowRadius * 2, shadowRadius * 2);
    const shadowMat = new THREE.MeshBasicMaterial({
      map: makeShadowTexture(),
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const shadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.position.set(center.x, box.min.y, center.z);
    scene.add(shadowPlane);
  },
  undefined,
  (error) => {
    console.error('Failed to load GLB:', error);
  }
);

// --- Resize ---

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- Render loop ---

function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  renderer.render(scene, camera);
}

animate();
