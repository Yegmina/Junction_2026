"""Explicit allow-list MCP adapter. Ground truth/scoring are never tools."""
from __future__ import annotations

import json
from .environment import Campaign


class AgentTools:
    """Small adapter also usable in trainers without the MCP dependency."""
    def __init__(self, campaign: Campaign):
        self._campaign = campaign

    def get_status(self) -> dict:
        return self._campaign.status()

    def action(self, action: str, arguments: dict | None = None) -> dict:
        return self._campaign.act(action,arguments)


def create_server(campaign: Campaign | None = None):
    try:
        try:
            from mcp.server.mcpserver import MCPServer as FastMCP, Image as MCPImage
        except ImportError:
            from mcp.server.fastmcp import FastMCP, Image as MCPImage
    except ImportError as exc:
        raise RuntimeError("MCP server requires the optional mcp package; install project dependencies") from exc
    campaign = campaign or Campaign()
    adapter = AgentTools(campaign)
    server = FastMCP("AeroRozum thermal survey")

    def respond(result):
        observation = result.get("observation") or result.get("crop")
        if observation and observation.get("image"):
            # Only paths produced by the campaign, never user-supplied paths.
            path = (campaign.output_dir/observation["image"]).resolve()
            if not path.is_relative_to(campaign.output_dir):
                raise ValueError("Observation artifact is outside data root")
            return [json.dumps(result),MCPImage(data=path.read_bytes(),format="png")]
        return result

    @server.tool()
    def get_status() -> dict:
        """Read current clock, camera pose, flight budgets and recorded evidence."""
        return adapter.get_status()

    @server.tool()
    def launch(altitude_m: float = 6):
        """Launch from base when turnaround has completed; consumes a sortie."""
        return respond(adapter.action("launch",{"altitude_m":altitude_m}))

    @server.tool()
    def move_to(x_m: float,y_m: float,altitude_m: float = 6,yaw_deg: float = 0):
        """Move to east/north metres at altitude; flight travel consumes time and battery."""
        return respond(adapter.action("move_to",{"x_m":x_m,"y_m":y_m,"altitude_m":altitude_m,"yaw_deg":yaw_deg}))

    @server.tool()
    def capture():
        """Capture a new camera observation at the actual current pose and time."""
        return respond(adapter.action("capture"))

    @server.tool()
    def inspect_observation(observation_id: str):
        """Inspect an existing recorded frame, consuming operator time."""
        return respond(adapter.action("inspect_observation",{"observation_id":observation_id}))

    @server.tool()
    def inspect_crop(observation_id: str,box: dict):
        """Inspect a recorded pixel crop using x_min/y_min/x_max/y_max in 0..1000."""
        return respond(adapter.action("inspect_crop",{"observation_id":observation_id,"box":box}))

    @server.tool()
    def wait_until(time_s: float):
        """Advance simulation time in seconds since midnight; airborne waiting uses battery."""
        return respond(adapter.action("wait_until",{"time_s":time_s}))

    @server.tool()
    def land():
        """Return to base, land, and start the battery turnaround interval."""
        return respond(adapter.action("land"))

    @server.tool()
    def record_candidate(evidence: list[dict],confidence: float = .5,notes: str = "",candidate_id: str | None = None):
        """Save or update a provisional candidate with recorded-frame normalized boxes."""
        arguments = {"evidence":evidence,"confidence":confidence,"notes":notes}
        if candidate_id:
            arguments["candidate_id"] = candidate_id
        return respond(adapter.action("record_candidate",arguments))

    @server.tool()
    def submit_report():
        """Finalize the evidence report. Hidden evaluation results are not returned."""
        return respond(adapter.action("submit_report"))

    return server


def main():
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
