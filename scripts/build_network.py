"""Generate the three-junction SUMO network and traffic demand."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenario"
GENERATED = SCENARIO / "generated"
CROSSING_LENGTH_METERS = 6.4
MIN_ALL_RED_SECONDS = 3


def _write_xml(path: Path, root: ElementTree.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def create_nodes() -> None:
    root = ElementTree.Element("nodes")
    junctions = {"J0": (0, 0), "J1": (200, 0), "J2": (400, 0)}
    for junction_id, (x, y) in junctions.items():
        node = ElementTree.SubElement(
            root,
            "node",
            id=junction_id,
            x=str(x),
            y=str(y),
            type="traffic_light",
        )
        incoming = {
            "north": f"N{junction_id[1]}_{junction_id}",
            "south": f"S{junction_id[1]}_{junction_id}",
        }
        if junction_id == "J0":
            incoming["west"] = "W0_J0"
            incoming["east"] = "J1_J0"
        elif junction_id == "J2":
            incoming["west"] = "J1_J2"
            incoming["east"] = "E2_J2"
        else:
            incoming["west"] = "J0_J1"
            incoming["east"] = "J2_J1"
        for edge_id in incoming.values():
            ElementTree.SubElement(
                root,
                "crossing",
                node=junction_id,
                edges=edge_id,
                width="3.0",
            )

    boundary = {
        "N0": (0, 100),
        "S0": (0, -100),
        "N1": (200, 100),
        "S1": (200, -100),
        "N2": (400, 100),
        "S2": (400, -100),
        "W0": (-100, 0),
        "E2": (500, 0),
    }
    for node_id, (x, y) in boundary.items():
        ElementTree.SubElement(root, "node", id=node_id, x=str(x), y=str(y))
    _write_xml(GENERATED / "network.nod.xml", root)


def create_edges() -> None:
    root = ElementTree.Element("edges")

    def edge(edge_id: str, source: str, target: str) -> None:
        ElementTree.SubElement(
            root,
            "edge",
            id=edge_id,
            **{"from": source, "to": target},
            numLanes="1",
            speed="13.9",
            sidewalkWidth="2.0",
        )

    for index in range(3):
        edge(f"N{index}_J{index}", f"N{index}", f"J{index}")
        edge(f"J{index}_N{index}", f"J{index}", f"N{index}")
        edge(f"S{index}_J{index}", f"S{index}", f"J{index}")
        edge(f"J{index}_S{index}", f"J{index}", f"S{index}")
    edge("W0_J0", "W0", "J0")
    edge("J0_W0", "J0", "W0")
    edge("E2_J2", "E2", "J2")
    edge("J2_E2", "J2", "E2")
    for left, right in (("J0", "J1"), ("J1", "J2")):
        edge(f"{left}_{right}", left, right)
        edge(f"{right}_{left}", right, left)
    _write_xml(GENERATED / "network.edg.xml", root)


def create_routes() -> None:
    root = ElementTree.Element("routes")
    vehicle_type = ElementTree.SubElement(
        root,
        "vType",
        id="car",
        accel="2.6",
        decel="4.5",
        sigma="0.5",
        length="5",
        minGap="2.5",
        maxSpeed="13.9",
    )
    vehicle_type.set("guiShape", "passenger")
    routes = {
        "corridor-east": "W0_J0 J0_J1 J1_J2 J2_E2",
        "corridor-west": "E2_J2 J2_J1 J1_J0 J0_W0",
        "north-south-0": "N0_J0 J0_S0",
        "south-north-0": "S0_J0 J0_N0",
        "north-south-1": "N1_J1 J1_S1",
        "south-north-1": "S1_J1 J1_N1",
        "north-south-2": "N2_J2 J2_S2",
        "south-north-2": "S2_J2 J2_N2",
    }
    for route_id, edges in routes.items():
        ElementTree.SubElement(root, "route", id=route_id, edges=edges)
        ElementTree.SubElement(
            root,
            "flow",
            id=f"flow-{route_id}",
            type="car",
            route=route_id,
            begin="0",
            end="3600",
            period="6",
            departLane="best",
            departSpeed="max",
        )
    _write_xml(GENERATED / "traffic.rou.xml", root)


def build(green_duration: int = 37, pedestrian_speed: float = 1.4) -> Path:
    if green_duration <= 0:
        raise ValueError("green_duration must be positive")
    if pedestrian_speed <= 0:
        raise ValueError("pedestrian_speed must be positive")
    netconvert = shutil.which("netconvert")
    if netconvert is None:
        raise RuntimeError(
            "netconvert was not found. Install SUMO and add its bin directory to PATH."
        )
    GENERATED.mkdir(parents=True, exist_ok=True)
    create_nodes()
    create_edges()
    create_routes()
    network = GENERATED / "network.net.xml"
    subprocess.run(
        [
            netconvert,
            "-n",
            str(GENERATED / "network.nod.xml"),
            "-e",
            str(GENERATED / "network.edg.xml"),
            "--crossings.guess",
            "true",
            "--sidewalks.guess",
            "true",
            "--sidewalks.guess.min-speed",
            "100",
            "-o",
            str(network),
        ],
        check=True,
    )
    tree = ElementTree.parse(network)
    for logic in tree.findall(".//tlLogic"):
        phases = logic.findall("phase")
        for index in (0, 3):
            phases[index].set("duration", str(green_duration))
        clearance = max(
            MIN_ALL_RED_SECONDS,
            math.ceil(CROSSING_LENGTH_METERS / pedestrian_speed),
        )
        for index in (2, 5):
            phases[index].set("duration", str(clearance))
    tree.write(network, encoding="utf-8", xml_declaration=True)
    config = ElementTree.Element("configuration")
    input_node = ElementTree.SubElement(config, "input")
    input_node.extend(
        [
            ElementTree.Element("net-file", value="network.net.xml"),
            ElementTree.Element("route-files", value="traffic.rou.xml"),
        ]
    )
    ElementTree.SubElement(config, "time-to-teleport", value="-1")
    ElementTree.SubElement(config, "begin", value="0")
    ElementTree.SubElement(config, "end", value="3600")
    _write_xml(GENERATED / "simulation.sumocfg", config)
    return network


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--green-duration", type=int, default=37)
    parser.add_argument("--pedestrian-speed", type=float, default=1.4)
    args = parser.parse_args()
    print(f"Generated SUMO scenario in {GENERATED}")
    build(args.green_duration, args.pedestrian_speed)
