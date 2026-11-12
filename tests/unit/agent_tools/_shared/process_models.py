"""Process models shared by the process-model and continuation tool tests."""

_BPMN = 'xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"'

# The spending-approval example of #241: one gateway-cycle loop and an XOR.
SPENDING_APPROVAL_MODEL = f"""<?xml version="1.0" encoding="UTF-8"?>
<definitions {_BPMN} id="defs">
  <process id="spending_approval">
    <startEvent id="start"/>
    <task id="open_requests" name="Open the requests page"/>
    <exclusiveGateway id="each_request" name="For each request_id in pending"/>
    <task id="open_request" name="Open the request"/>
    <exclusiveGateway id="decide"/>
    <task id="click_approve" name="Click Approve"/>
    <task id="click_reject" name="Click Reject"/>
    <exclusiveGateway id="decide_join"/>
    <task id="back_to_list" name="Return to the list"/>
    <exclusiveGateway id="each_request_next"/>
    <endEvent id="end"/>
    <sequenceFlow id="f1" sourceRef="start" targetRef="open_requests"/>
    <sequenceFlow id="f2" sourceRef="open_requests" targetRef="each_request"/>
    <sequenceFlow id="f3" sourceRef="each_request" targetRef="open_request"/>
    <sequenceFlow id="f4" sourceRef="open_request" targetRef="decide"/>
    <sequenceFlow id="f5" sourceRef="decide" targetRef="click_approve" name="amount within limit"/>
    <sequenceFlow id="f6" sourceRef="decide" targetRef="click_reject" name="otherwise"/>
    <sequenceFlow id="f7" sourceRef="click_approve" targetRef="decide_join"/>
    <sequenceFlow id="f8" sourceRef="click_reject" targetRef="decide_join"/>
    <sequenceFlow id="f9" sourceRef="decide_join" targetRef="back_to_list"/>
    <sequenceFlow id="f10" sourceRef="back_to_list" targetRef="each_request_next"/>
    <sequenceFlow id="f11" sourceRef="each_request_next" targetRef="each_request" name="next request_id"/>
    <sequenceFlow id="f12" sourceRef="each_request_next" targetRef="end" name="all pending done"/>
  </process>
</definitions>"""

# An XOR without loops, with a namespace prefix as bpmn-js exports it.
APPROVAL_XOR_MODEL = """<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="approval">
    <bpmn:startEvent id="start"/>
    <bpmn:task id="open_request" name="Open the request"/>
    <bpmn:task id="read_amount" name="Read the amount"/>
    <bpmn:exclusiveGateway id="within_limit" name="Amount within limit?"/>
    <bpmn:task id="click_approve" name="Click Approve"/>
    <bpmn:task id="confirm_approval" name="Confirm the approval"/>
    <bpmn:task id="click_reject" name="Click Reject"/>
    <bpmn:task id="enter_reason" name="Enter the reason"/>
    <bpmn:exclusiveGateway id="within_limit_join"/>
    <bpmn:task id="save_request" name="Save the request"/>
    <bpmn:endEvent id="end"/>
    <bpmn:sequenceFlow id="f1" sourceRef="start" targetRef="open_request"/>
    <bpmn:sequenceFlow id="f2" sourceRef="open_request" targetRef="read_amount"/>
    <bpmn:sequenceFlow id="f3" sourceRef="read_amount" targetRef="within_limit"/>
    <bpmn:sequenceFlow id="f4" sourceRef="within_limit" targetRef="click_approve"/>
    <bpmn:sequenceFlow id="f5" sourceRef="click_approve" targetRef="confirm_approval"/>
    <bpmn:sequenceFlow id="f6" sourceRef="within_limit" targetRef="click_reject"/>
    <bpmn:sequenceFlow id="f7" sourceRef="click_reject" targetRef="enter_reason"/>
    <bpmn:sequenceFlow id="f8" sourceRef="confirm_approval" targetRef="within_limit_join"/>
    <bpmn:sequenceFlow id="f9" sourceRef="enter_reason" targetRef="within_limit_join"/>
    <bpmn:sequenceFlow id="f10" sourceRef="within_limit_join" targetRef="save_request"/>
    <bpmn:sequenceFlow id="f11" sourceRef="save_request" targetRef="end"/>
  </bpmn:process>
</bpmn:definitions>"""

# Approvers (outer loop) and each approver's requests (inner loop), as gateway cycles.
NESTED_LOOPS_MODEL = f"""<definitions {_BPMN}>
  <process id="approvals">
    <startEvent id="start"/>
    <task id="log_in" name="Log in"/>
    <exclusiveGateway id="each_approver" name="For each approver"/>
    <task id="open_approver" name="Open the approver"/>
    <exclusiveGateway id="each_request" name="For each request of the approver"/>
    <task id="open_request" name="Open the request"/>
    <exclusiveGateway id="decide"/>
    <task id="click_approve" name="Click Approve"/>
    <task id="click_reject" name="Click Reject"/>
    <exclusiveGateway id="decide_join"/>
    <task id="back_to_list" name="Return to the list"/>
    <exclusiveGateway id="each_request_next"/>
    <exclusiveGateway id="each_approver_next"/>
    <endEvent id="end"/>
    <sequenceFlow id="f1" sourceRef="start" targetRef="log_in"/>
    <sequenceFlow id="f2" sourceRef="log_in" targetRef="each_approver"/>
    <sequenceFlow id="f3" sourceRef="each_approver" targetRef="open_approver"/>
    <sequenceFlow id="f4" sourceRef="open_approver" targetRef="each_request"/>
    <sequenceFlow id="f5" sourceRef="each_request" targetRef="open_request"/>
    <sequenceFlow id="f6" sourceRef="open_request" targetRef="decide"/>
    <sequenceFlow id="f7" sourceRef="decide" targetRef="click_approve"/>
    <sequenceFlow id="f8" sourceRef="decide" targetRef="click_reject"/>
    <sequenceFlow id="f9" sourceRef="click_approve" targetRef="decide_join"/>
    <sequenceFlow id="f10" sourceRef="click_reject" targetRef="decide_join"/>
    <sequenceFlow id="f11" sourceRef="decide_join" targetRef="back_to_list"/>
    <sequenceFlow id="f12" sourceRef="back_to_list" targetRef="each_request_next"/>
    <sequenceFlow id="f13" sourceRef="each_request_next" targetRef="each_request"/>
    <sequenceFlow id="f14" sourceRef="each_request_next" targetRef="each_approver_next"/>
    <sequenceFlow id="f15" sourceRef="each_approver_next" targetRef="each_approver"/>
    <sequenceFlow id="f16" sourceRef="each_approver_next" targetRef="end"/>
  </process>
</definitions>"""

# The requests loop as a looping subprocess instead of a gateway cycle.
LOOPING_SUBPROCESS_MODEL = f"""<definitions {_BPMN}>
  <process id="spending_approval">
    <startEvent id="start"/>
    <task id="open_requests" name="Open the requests page"/>
    <subProcess id="each_request" name="For each request">
      <standardLoopCharacteristics/>
      <startEvent id="sub_start"/>
      <task id="open_request" name="Open the request"/>
      <task id="click_approve" name="Click Approve"/>
      <endEvent id="sub_end"/>
      <sequenceFlow id="s1" sourceRef="sub_start" targetRef="open_request"/>
      <sequenceFlow id="s2" sourceRef="open_request" targetRef="click_approve"/>
      <sequenceFlow id="s3" sourceRef="click_approve" targetRef="sub_end"/>
    </subProcess>
    <endEvent id="end"/>
    <sequenceFlow id="f1" sourceRef="start" targetRef="open_requests"/>
    <sequenceFlow id="f2" sourceRef="open_requests" targetRef="each_request"/>
    <sequenceFlow id="f3" sourceRef="each_request" targetRef="end"/>
  </process>
</definitions>"""

# Out of scope (#231): a parallel multi-instance subprocess.
PARALLEL_MULTI_INSTANCE_MODEL = LOOPING_SUBPROCESS_MODEL.replace(
    "<standardLoopCharacteristics/>", "<multiInstanceLoopCharacteristics/>"
)

# A cycle between `a` and `b` entered at both: no single loop header.
IRREDUCIBLE_MODEL = f"""<definitions {_BPMN}>
  <process id="p">
    <startEvent id="start"/>
    <exclusiveGateway id="split"/>
    <task id="a" name="A"/>
    <task id="b" name="B"/>
    <endEvent id="end"/>
    <sequenceFlow id="f1" sourceRef="start" targetRef="split"/>
    <sequenceFlow id="f2" sourceRef="split" targetRef="a"/>
    <sequenceFlow id="f3" sourceRef="split" targetRef="b"/>
    <sequenceFlow id="f4" sourceRef="a" targetRef="b"/>
    <sequenceFlow id="f5" sourceRef="b" targetRef="a"/>
    <sequenceFlow id="f6" sourceRef="a" targetRef="end"/>
  </process>
</definitions>"""

# The bridge's compact outline of the spending-approval process.
COMPACT_MODEL = """process spending_approval
  open_requests "Open the requests page"
  for each_request: request_id in pending
    open_request "Open the request"
    if decide: amount within limit
      click_approve "Click Approve"
    otherwise
      click_reject "Click Reject"
    back_to_list "Return to the list"
"""
