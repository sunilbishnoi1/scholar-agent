import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, MeshDistortMaterial, Environment, Float } from '@react-three/drei';
import * as THREE from 'three';

// Agent Node Component
const AgentNode: React.FC<{
  position: [number, number, number];
  color: string;
  delay: number;
}> = ({ position, color, delay }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = state.clock.elapsedTime * 0.2 + delay;
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.3 + delay;
    }
  });

  return (
    <Float speed={1.5} rotationIntensity={0.5} floatIntensity={0.5}>
      <group position={position}>
        <Sphere ref={meshRef} args={[0.3, 32, 32]}>
          <MeshDistortMaterial
            color={color}
            attach="material"
            distort={0.3}
            speed={2}
            roughness={0.1}
            metalness={0.8}
          />
        </Sphere>
        {/* Connection lines */}
        <mesh>
          <sphereGeometry args={[0.35, 16, 16]} />
          <meshStandardMaterial
            color={color}
            transparent
            opacity={0.2}
            emissive={color}
            emissiveIntensity={0.5}
          />
        </mesh>
      </group>
    </Float>
  );
};

// Connection Line Component
const ConnectionLine: React.FC<{
  start: [number, number, number];
  end: [number, number, number];
  color: string;
}> = ({ start, end, color }) => {
  const points = useMemo(
    () => [
      new THREE.Vector3(...start),
      new THREE.Vector3(...end),
    ],
    [start, end]
  );

  const curve = new THREE.CatmullRomCurve3(points);

  return (
    <mesh>
      <tubeGeometry args={[curve, 20, 0.02, 8, false]} />
      <meshStandardMaterial
        color={color}
        transparent
        opacity={0.4}
        emissive={color}
        emissiveIntensity={0.3}
      />
    </mesh>
  );
};

// Main Scene Component
const Scene: React.FC = () => {
  const agents = useMemo(
    () => [
      {
        position: [-1.5, 1, 0] as [number, number, number],
        color: '#00F5C8', // Aurora Teal - Planner
      },
      {
        position: [1.5, 1, 0] as [number, number, number],
        color: '#00B894', // Discovery Green - Retriever
      },
      {
        position: [0, -1, 0] as [number, number, number],
        color: '#FFB900', // Insight Amber - Analyzer
      },
      {
        position: [-1, -0.5, 1.5] as [number, number, number],
        color: '#A1A1AA', // Obsidian neutral - Quality Checker
      },
      {
        position: [1, -0.5, -1.5] as [number, number, number],
        color: '#008C67', // Deep Aurora Teal - Synthesizer
      },
    ],
    []
  );

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#00B894" />

      {agents.map((agent, index) => (
        <AgentNode
          key={index}
          position={agent.position}
          color={agent.color}
          delay={index * 0.5}
        />
      ))}

      {/* Connection lines between agents */}
      <ConnectionLine
        start={agents[0].position}
        end={agents[1].position}
        color="#00F5C8"
      />
      <ConnectionLine
        start={agents[1].position}
        end={agents[2].position}
        color="#00B894"
      />
      <ConnectionLine
        start={agents[2].position}
        end={agents[0].position}
        color="#FFB900"
      />
      <ConnectionLine
        start={agents[3].position}
        end={agents[2].position}
        color="#A1A1AA"
      />
      <ConnectionLine
        start={agents[4].position}
        end={agents[2].position}
        color="#008C67"
      />

      <OrbitControls
        enableZoom={false}
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.5}
        minPolarAngle={Math.PI / 3}
        maxPolarAngle={Math.PI / 2.2}
      />
      <Environment preset="sunset" />
    </>
  );
};

export const AnimatedWorkflow: React.FC = () => {
  return (
    <div
      className="w-full h-full rounded-[var(--radius-xl)] overflow-hidden"
      style={{
        background: 'transparent',
      }}
    >
      <Canvas
        camera={{ position: [0, 0, 5], fov: 50 }}
        style={{ background: 'transparent' }}
        gl={{ antialias: true, alpha: true }}
      >
        <Scene />
      </Canvas>
    </div>
  );
};
