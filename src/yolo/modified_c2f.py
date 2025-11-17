class C2f(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

        # Always set attribute to avoid AttributeError in forward
        self.liam = None
        try:
            self.liam = LIAM(c2)
        except Exception:
            self.liam = None

        # container to keep last spatial attention map for visualization/debugging
        self.last_attn_map = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with CBAM applied to final output (if available)."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))

        # apply CBAM if available; store spatial map in self.last_attn_map (or None)
        if getattr(self, 'liam', None) is not None:
            try:
                out_liam = self.liam(out, return_map=True)
                if isinstance(out_liam, tuple) and len(out_liam) == 2:
                    out, spat = out_liam
                    self.last_attn_map = spat.detach() if spat is not None else None
                else:
                    out = out_liam
                    self.last_attn_map = None
            except Exception:
                self.last_attn_map = None

        return out

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk(). Also applies CBAM to final output."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))

        if getattr(self, 'liam', None) is not None:
            try:
                out_liam = self.cbam(out, return_map=True)
                if isinstance(out_liam, tuple) and len(out_liam) == 2:
                    out, spat = out_liam
                    self.last_attn_map = spat.detach() if spat is not None else None
                else:
                    out = out_liam
                    self.last_attn_map = None
            except Exception:
                self.last_attn_map = None

        return out
