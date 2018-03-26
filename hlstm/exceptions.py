
class VariableNotFoundException(Exception):
    def __init__(self, variable, where, msg=''):
        self.variable = variable
        self.where = where
        self.msg = msg

    def __str__(self):
        message = "variable %s not found in %s" % (self.variable, self.where)
        if self.msg:
            message +='\n'+self.msg



