class SortedPersistentTileScheduler:
    """KAI H200: tiles are dispatched in a host-provided order (block-sparse tiles sorted by list length, longest
    first). order[t] encodes tile t as (batch * num_head + head) * num_block + m_block (int32).

    Default: one CTA per tile (grid = total tiles) launched in sorted order. The hardware dispatches CTAs
    roughly by blockIdx, which gives LPT behaviour; the tile validity is a compile-time constant exactly as
    in SingleTileScheduler, so the kernel's tile loop stays a single straight-line iteration (a dynamic loop
    makes the 240-register consumers spill on SM90). KAI_SORTED_PERSISTENT=1 switches to a persistent grid of
    SM-count CTAs with a dynamic loop (experimental: correct only with a delayed Q release, and slow).
    """

    @dataclass
    class Params(ParamsBase):
        order: cute.Tensor
        total_tiles: Int32
        num_block_divmod: FastDivmodDivisorV2
        num_head_divmod: FastDivmodDivisorV2
        persistent: cutlass.Constexpr[bool] = False

        @staticmethod
        def create(
            args: TileSchedulerArguments, order: cute.Tensor, *, loc=None, ip=None
        ) -> "SortedPersistentTileScheduler.Params":
            import os as _os
            total_tiles = args.num_block * args.num_head * args.num_batch
            return SortedPersistentTileScheduler.Params(
                order,
                total_tiles,
                FastDivmodDivisorV2(args.num_block),
                FastDivmodDivisorV2(args.num_head),
                persistent=_os.environ.get("KAI_SORTED_PERSISTENT", "0") == "1",
            )

    def __init__(self, params: Params, tile_pos: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_pos = tile_pos
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        order: cute.Tensor,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        return SortedPersistentTileScheduler.Params.create(args, order, loc=loc, ip=ip)

    @staticmethod
    def create(
        params: Params, ctx: SchedulerState | None = None, *, loc=None, ip=None
    ) -> "SortedPersistentTileScheduler":
        tile_pos = cute.arch.block_idx()[0]
        return SortedPersistentTileScheduler(params, tile_pos, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        if const_expr(not params.persistent):
            return (params.total_tiles, Int32(1), Int32(1))
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        return (cutlass.min(Int32(sm_count), params.total_tiles), Int32(1), Int32(1))

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(not self.params.persistent):
            is_valid = self._is_first_block            # compile-time: one tile per CTA
            pos = self._tile_pos
        else:
            is_valid = self._tile_pos < self.params.total_tiles
            pos = cutlass.min(self._tile_pos, self.params.total_tiles - 1)
        tile_id = Int32(self.params.order[pos])
        hn_idx, block_idx = divmod(tile_id, self.params.num_block_divmod)
        batch_idx, head_idx = divmod(hn_idx, self.params.num_head_divmod)
        return WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(not self.params.persistent):
            self._is_first_block = False
        else:
            self._tile_pos += cute.arch.grid_dim()[0]
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_pos]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_pos], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        scheduler = SortedPersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)
        scheduler._is_first_block = self._is_first_block
        return scheduler


