classdef SemiCircular
   properties
      Radius
   end
   properties (Constant)
      P = pi
   end
   properties (Dependent)
      Area
   end
   methods
      %%% Init
      function obj = SemiCircular(r)
         if nargin > 0
            obj.Radius = r;
         end
      end
      function val = get.Area(self)
         val = self.P*self.Radius^2;
      end
      function self = set.Radius(self,val)
         if val < 0
            error('Radius must be positive')
         end
         self.Radius = val;
      end
      %%% End of init
   end
   methods (Static)
      function self = init(radius)
         self = SemiCircular(radius);
      end
      function [y1, y2] = test(self, x)
          y1 = x
          y2 = self.Radius
      end
   end
end