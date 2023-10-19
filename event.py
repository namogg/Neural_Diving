from pyscipopt import  Eventhdlr, SCIP_EVENTTYPE, SCIP_STATUS, SCIP_PARAMSETTING,SCIP_STAGE
class MyEvent(Eventhdlr):
    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)

    def eventexit(self):
        return
        
    def eventexec(self, event):
        self.model.interruptSolve()

class NodeEvent(Eventhdlr):
    def eventinit(self):
        self.gaps = []
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)
        
    def eventexit(self):
        return

    def eventexec(self, event):
        self.gaps.append(self.model.getGap())