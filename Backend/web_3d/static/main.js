import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const URL = '/api/clench';
let mixer, actionClench, clock = new THREE.Clock();

// scene & camera
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.1, 100);
camera.position.set(0, 0.5, 2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

// lights
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(2, 2, 2);
scene.add(dir);

// loader
const loader = new GLTFLoader();
loader.load('/static/hand.glb', gltf => {
    const model = gltf.scene;
    scene.add(model);
    mixer = new THREE.AnimationMixer(model);
    actionClench = mixer.clipAction(gltf.animations[0]);  // 0 = clench
    actionClench.clampWhenFinished = true;
    actionClench.loop = THREE.LoopOnce;
    animate();
});

// polling loop
let target = 0, current = 0;
setInterval(async () => {
    const res = await fetch(URL);
    const data = await res.json();
    target = data.clench ? 1 : 0;
}, 150);

// smooth blend
function animate() {
    requestAnimationFrame(animate);
    const dt = clock.getDelta();
    current += (target - current) * dt * 12;
    if (mixer) mixer.setTime(current);
    renderer.render(scene, camera);
}

// resize
addEventListener('resize', () => {
    camera.aspect = innerWidth/innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
});