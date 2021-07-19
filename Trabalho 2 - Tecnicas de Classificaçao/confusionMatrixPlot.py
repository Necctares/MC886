y_pred = np.array([])
for i in saida:
  y_pred = np.append(y_saida,np.argmax(i))

y_verdade = np.array([])
for i in y_teste:
  y_verdade = np.append(y_verdade,np.argmax(i))


labels = ['camiseta','calca','pulover','vestido','casaco','sandalia','camisa','sapato','bolsa','bota']
c_m = confusion_matrix(y_verdade, y_saida)

figura = plt.figure(figsize=(10,9))
axis = figura.add_subplot(111)
caxis = axis.matshow(c_m)
plt.title('Matriz de Confusão: Rede Neural Simples (item 2)')
figura.colorbar(caxis)
plt.xticks([0,1,2,3,4,5,6,7,8,9],labels)
plt.yticks([0,1,2,3,4,5,6,7,8,9],labels)
plt.xlabel('Predição')
plt.ylabel('Esperado')
plt.show()