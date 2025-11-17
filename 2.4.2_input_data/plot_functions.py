
import folium
from folium.plugins import TimestampedGeoJson
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


def create_animated_events_map(
    events_df,
    cells_df,
    periods_to_fade_out=12,
    period_duration_min=5,
    zoom_start=11,
    tiles="CartoDB positron",
    max_speed=4
):
    """
    Create an animated folium map showing events fading out over time.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataset with columns: user_id, timestamp, cell_id
    cells_df : pd.DataFrame
        Cells dataset with columns: cell_id, longitude, latitude
    periods_to_fade_out : int, default=12
        Number of time periods until events disappear completely
        (default 12 periods x 5 min = 1 hour)
    period_duration_min : int, default=5
        Duration of each period in minutes
    zoom_start : int, default=11
        Initial zoom level for the map
    tiles : str, default="CartoDB positron"
        Map tile style
    max_speed : int, default=4
        Maximum animation speed
        
    Returns
    -------
    folium.Map
        Interactive map with animated, fading events
    """
    # 1) Join coordinates to events
    ev = events_df.merge(cells_df, on="cell_id", how="left")
    
    # 2) Expand events over time with fading strength
    ev = (
        ev.loc[ev.index.repeat(periods_to_fade_out)]
          .assign(step=lambda x: x.groupby(level=0).cumcount())
          .assign(
              timestamp=lambda x: x["timestamp"] + pd.to_timedelta(
                  x["step"] * period_duration_min, unit="min"
              ),
              strength=lambda x: 100 - x["step"] * 100 / periods_to_fade_out
          )
          .drop(columns="step")
          .reset_index(drop=True)
    ).sort_values(by=['user_id', 'timestamp'])
    
    # 3) Build GeoJSON features with temporal properties
    features = []
    for _, r in ev.iterrows():
        t = r["timestamp"]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["longitude"]), float(r["latitude"])],
            },
            "properties": {
                "time": pd.Timestamp(t).isoformat(),
                "popup": f"cell_id={r.get('cell_id','')} | strength={r['strength']:.1f}",
                "icon": "circle",
                "iconstyle": {
                    "radius": float(r["strength"] / 3),
                    "fill": True,
                    "fillColor": "#3388ff",
                    "fillOpacity": r["strength"] / 100,
                    "stroke": False,
                    "weight": 0
                },
            },
        })
    
    geojson = {"type": "FeatureCollection", "features": features}
    
    # 4) Create map with time slider
    center = [ev["latitude"].mean(), ev["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles)
    
    # Calculate duration (slightly less than period to avoid overlap)
    duration_min = period_duration_min - 1
    duration_sec = 59
    duration_str = f"PT{duration_min}M{duration_sec}S"
    
    TimestampedGeoJson(
        data=geojson,
        period=f"PT{period_duration_min}M",
        duration=duration_str,
        add_last_point=False,
        auto_play=False,
        loop_button=True,
        time_slider_drag_update=True,
        date_options="YYYY-MM-DD HH:mm:ss",
        max_speed=max_speed
    ).add_to(m)
    
    return m


def plot_3d_trajectory(
    df,
    cells_df=None,
    output_file=None,
    color_by_tag=None,
    title=None
):
    """
    Create a 3D trajectory plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: timestamp, x, y, cell_id, and speed column
    cells_df : pd.DataFrame
        DataFrame with columns: latitude, longitude, cell_id
    output_file : str, optional
        If provided, save the plot to this HTML file path
    color_by_tag : str, optional
        Column to color by
    title : str, optional
        Custom title for the plot
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D trajectory plot
    """
    # Prepare data
    df = df.copy()

    if cells_df is not None and 'x' not in df.columns:
        df = df.merge(cells_df, how='left', on=['cell_id'])
        df['x'] = df['latitude']
        df['y'] = df['longitude']

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    scatter_kwargs = dict(
        data_frame=df,
        x="x",
        y="y",
        z="timestamp",
        hover_data=["cell_id"]
    )
    
    if color_by_tag is not None and color_by_tag in df.columns:
        scatter_kwargs["color"] = color_by_tag

    # Main 3D scatter
    fig = px.scatter_3d(**scatter_kwargs)

    # Ensure markers (no lines)
    for tr in fig.data:
        tr.mode = "markers"

    # Add trajectory line
    fig.add_trace(go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["timestamp"],
        mode="lines",
        line=dict(color="gray", width=3),
        name="trajectory",
        showlegend=True
    ))

    # Format z-axis (time) ticks
    tickvals = df["timestamp"].dt.floor("h").unique()[::3]
    ticktext = [t.strftime("%H:%M") for t in tickvals]

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Hour",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(tickvals=tickvals, ticktext=ticktext),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=title or "3D trajectory",
        showlegend=True
    )

    # Save to file if requested
    if output_file:
        fig.write_html(output_file, include_plotlyjs="cdn", full_html=True)
        print(f"Saved plot to: {output_file}")

    return fig


def plot_speed_over_time(
    df,
    speed_col="speed",
    scale_mode="log",
    figsize=(12, 5),
    cmap="viridis",
    marker="s",
    marker_size=35,
    alpha=0.85,
    title=None
):
    """
    Plot speed values over time with color-coded markers.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: user_id, timestamp, and speed column
    speed_col : str, default="speed"
        Name of the speed column to plot
    scale_mode : str, default="log"
        Transformation for color scale: "linear", "log", or "sqrt"
    figsize : tuple, default=(12, 5)
        Figure size (width, height) in inches
    cmap : str, default="viridis"
        Matplotlib colormap name
    marker : str, default="s"
        Marker style (e.g., "o", "s", "^")
    marker_size : int, default=35
        Size of markers
    alpha : float, default=0.85
        Marker transparency (0.0 to 1.0)
    title : str, optional
        Custom plot title
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    # Prepare data
    df = df.sort_values(["timestamp"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
        
    plot_df = df.copy()
        
    def transform_speed(x, mode="log"):
        """Transform speed values for color scaling."""
        if mode == "log":
            return np.log10(1 + np.maximum(x, 0))
        elif mode == "sqrt":
            return np.sqrt(np.maximum(x, 0))
        else:
            return x
    
    # Transform speed for color scaling
    plot_df["speed_scaled"] = transform_speed(plot_df[speed_col], scale_mode)
    
    # Colormap setup
    cmap_obj = plt.get_cmap(cmap)
    vmin = np.nanpercentile(plot_df["speed_scaled"], 1)
    vmax = np.nanpercentile(plot_df["speed_scaled"], 99)
    
    # Handle edge case where all values are the same
    if vmin == vmax:
        vmin = max(0.0, vmin - 1.0)
        vmax = vmax + 1.0
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sc = ax.scatter(
        plot_df["timestamp"],
        plot_df[speed_col],
        c=plot_df["speed_scaled"],
        cmap=cmap_obj,
        norm=norm,
        s=marker_size,
        marker=marker,
        alpha=alpha,
        label=f"{speed_col} (prev cell → next cell)"
    )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    
    # Colorbar tick labels mapped back to real speeds
    if scale_mode == "log":
        ticks = np.linspace(vmin, vmax, 6)
        tick_labels = [f"{10**t - 1:.1f}" for t in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
    
    cbar.set_label(f"{speed_col.capitalize()} (km/h) [{scale_mode} scale]")
    
    # Labels, grid, legend
    plot_title = title or f"Speed over time"
    ax.set_title(plot_title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Speed (km/h)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    
    return fig, ax


def plot_3d_trajectory_with_speed(
    df,
    speed_col="speed",
    scale_mode="log",
    output_file=None,
    title=None
):
    """
    Create a 3D trajectory plot colored by speed values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: timestamp, latitude, longitude, cell_id, and speed column
    speed_col : str, default="speed"
        Name of the speed column to use for coloring
    scale_mode : str, default="log"
        Transformation to apply to speed values: "linear", "log", or "sqrt"
    output_file : str, optional
        If provided, save the plot to this HTML file path
    title : str, optional
        Custom title for the plot
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D trajectory plot
    """
    # Prepare data
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    def transform_speed(series, mode="log"):
        """Transform speed values for better visualization."""
        arr = series.to_numpy(dtype=float)
        if mode == "log":
            # log10(1 + x) to keep 0 valid and compress high values
            return np.log10(1 + np.maximum(arr, 0))
        elif mode == "sqrt":
            return np.sqrt(np.maximum(arr, 0))
        else:
            return arr
    
    # Transform speed values
    transformed = transform_speed(df[speed_col], scale_mode)
    df = df.assign(speed_transformed=transformed)
    
    # Separate finite and NaN values
    finite_mask = np.isfinite(df["speed_transformed"].to_numpy(dtype=float))
    df_finite = df.loc[finite_mask].copy()
    df_nan = df.loc[~finite_mask].copy()
    
    # Compute color range from finite values
    if len(df_finite) > 0:
        vmin = float(np.nanpercentile(df_finite["speed_transformed"], 1))
        vmax = float(np.nanpercentile(df_finite["speed_transformed"], 99))
        if vmin == vmax:
            vmin = max(0.0, vmin - 1.0)
            vmax = vmax + 1.0
    else:
        vmin, vmax = 0.0, 1.0
    
    # Create main scatter plot with finite speed values
    fig = px.scatter_3d(
        df_finite,
        x="x",
        y="y",
        z="timestamp",
        color="speed_transformed",
        color_continuous_scale="Viridis",
        range_color=(vmin, vmax),
        hover_data=["cell_id", speed_col],
    )
    
    for tr in fig.data:
        tr.mode = "markers"
    
    # Add NaN markers in gray
    if len(df_nan) > 0:
        fig.add_trace(go.Scatter3d(
            x=df_nan["x"],
            y=df_nan["y"],
            z=df_nan["timestamp"],
            mode="markers",
            marker=dict(size=5, color="lightgray"),
            name=f"{speed_col} = NaN",
            showlegend=True,
        ))
    
    # Add trajectory line
    fig.add_trace(go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["timestamp"],
        mode="lines",
        line=dict(color="gray", width=3),
        name="trajectory",
        showlegend=True
    ))
    
    # Format z-axis (time) with hourly ticks
    tickvals = df["timestamp"].dt.floor("h").unique()[::3]
    ticktext = [t.strftime("%H:%M") for t in tickvals]
    
    # Create colorbar tick labels (inverse transform for log scale)
    if scale_mode == "log":
        colorbar_ticktext = [f"{10**(t)-1:.1f}" for t in np.linspace(vmin, vmax, 5)]
    else:
        colorbar_ticktext = [f"{t:.1f}" for t in np.linspace(vmin, vmax, 5)]
    
    # Update layout
    plot_title = title or f"3D trajectory: markers by {speed_col} ({scale_mode} scale)"
    
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Hour",
            zaxis=dict(tickvals=tickvals, ticktext=ticktext),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=plot_title,
        coloraxis_colorbar=dict(
            title=f"{speed_col} ({scale_mode} scale)",
            tickvals=np.linspace(vmin, vmax, 5),
            ticktext=colorbar_ticktext,
        ),
        showlegend=False
    )
    
    # Save to file if requested
    if output_file:
        fig.write_html(output_file, include_plotlyjs="cdn", full_html=True)
        print(f"Saved plot to: {output_file}")
    
    return fig
