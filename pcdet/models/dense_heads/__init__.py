from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_vote import PointHeadVote
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .IASSD_head import IASSD_Head
from .anchor_head_rdiou import AnchorHeadRDIoU
from .anchor_head_rdiou_3cat import AnchorHeadRDIoU_3CAT


__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'IASSD_Head': IASSD_Head,
    'PointHeadVote': PointHeadVote,
    'AnchorHeadRDIoU': AnchorHeadRDIoU,
    'AnchorHeadRDIoU_3CAT': AnchorHeadRDIoU_3CAT,
}
