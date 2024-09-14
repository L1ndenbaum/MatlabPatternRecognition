x11 = 1:10;
y11 = x11;
x12 = 1:10;
y12 = 2*x11;

x21 = 0:0.01:2*pi;
y21 = sin(x21);
y22 = cos(x21);
subplot(2,2,1)
plot(x11, y11, 'ro',x12, y12, 'bo')
xlabel('x1')
ylabel('y1')
title('scatter')
legend('y=x', 'y=2*x')
subplot(2,2,2)
plot(x21, y21, 'r-',x21, y22, 'b-')
xlabel('x2')
ylabel('y2')
title('trigonometric')
legend('y=sinx', 'y=cosx')
subplot(2,2,3)
plot(x11, y11, 'b.',x12, y12, 'g.')
xlabel('x3')
ylabel('y3')
title('scatter, different color')
legend('y=x', 'y=2*x')
subplot(2,2,4)
plot(x21, y21, 'b*',x21, y22, 'r*')
xlabel('x4')
ylabel('y4')
title('trigonometric, different line type')
legend('y=sinx', 'y=cosx')